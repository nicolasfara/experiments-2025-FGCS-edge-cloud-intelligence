package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{AllocatorProperty, BatteryEquippedDevice, Environment, Node, PayPerUseDevice, Position, TimeDistribution}
import it.unibo.alchemist.utils.AlchemistScafiUtils.getAlchemistActions
import it.unibo.alchemist.utils.Molecules
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import learning.model.{ActionSpace, Cloud, Component, EdgeServer, MySelf, PairComponentDevice}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.{Reader, Writer}

import scala.jdk.CollectionConverters.CollectionHasAsScala

class EvalDensity [T, P <: Position[P]](
  environment: Environment[T, P],
  distribution: TimeDistribution[T],
  seed: Int
) extends AbstractGlobalReaction[T, P](environment, distribution) {

  private implicit def toMolecule(name: String): SimpleMolecule = new SimpleMolecule(name)
  private implicit def toPythonCollection[K: Reader: Writer](collection: Seq[K]): py.Any = collection.toPythonProxy
  private implicit def toPythonCollectionNested[K: Reader: Writer](collection: Seq[Seq[K]]): py.Any = collection.toPythonProxy

  private lazy val cloudNodes: Seq[Node[T]] = nodes
    .filter(n => n.contains(Molecules.cloud))
    .sortBy(node => node.getId)

  private lazy val infrastructuralNodes: Seq[Node[T]] = nodes
    .filter(n => n.contains(Molecules.infrastructural))
    .sortBy(node => node.getId)

  private lazy val applicationNodes: Seq[Node[T]] = nodes
    .filterNot(n => n.contains(Molecules.infrastructural))
    .filterNot(n => n.contains(Molecules.cloud))
    .sortBy(node => node.getId)

  private lazy val components: Seq[Component] = getComponents

  private var executedToHetero = false

  private lazy val actionSpace = ActionSpace(components, infrastructuralNodes.size, cloudNodes.size)

  private val learner = rlUtils.DQNTrainer(36, seed, 1000, 400)


  private def createGraph(): py.Dynamic = {

    val adjacencyAppToApp = getEdgeIndexes(applicationNodes.map(_.getId), applicationNodes.map(_.getId))
    val infraNodes = infrastructuralNodes.appendedAll(cloudNodes)
    val adjacencyInfraToInfra = getEdgeIndexes(infraNodes.map(_.getId), infraNodes.map(_.getId))
    val adjacencyAppToInfra = getEdgeIndexesAll(applicationNodes.map(_.getId), infraNodes.map(_.getId)) //- applicationNodes.size))
    val pyAdjacencyAppToApp = toTorchTensor(adjacencyAppToApp)
    val pyAdjacencyInfraToInfra = toTorchTensor(scaleIds(adjacencyInfraToInfra))
    val pyAdjacencyAppToInfra = toTorchTensor(scaleIds(adjacencyAppToInfra, false))
    val pyFeaturesApplication = toFeatures(applicationNodes)
    val pyFeaturesInfrastructural = toFeatures(infraNodes)
    val featureAppToApp = createFeatureTensor(adjacencyAppToApp)
    val featureInfraToInfra = createFeatureTensor(adjacencyInfraToInfra)
    val featureAppToInfra = createFeatureTensor(adjacencyAppToInfra)

    rlUtils.create_graph(
      pyFeaturesApplication,
      pyFeaturesInfrastructural,
      pyAdjacencyAppToInfra,
      pyAdjacencyAppToApp,
      pyAdjacencyInfraToInfra,
      featureAppToInfra,
      featureAppToApp,
      featureInfraToInfra,
    )
  }

  override protected def executeBeforeUpdateDistribution(): Unit = handleGraph(createGraph())

  protected def handleGraph(observation: py.Dynamic): Unit = {
    if (!executedToHetero) {
      learner.toHetero(observation)
      learner.load_model_from_snapshot("networks/model-global-round-101-seed-2")
      executedToHetero = true
    }

    val actions = learner.select_action(observation, 0.0)

    actions
      .tolist()
      .as[List[Int]]
      .zipWithIndex
      .foreach { case (actionIndex, nodeIndex) =>
        val node = applicationNodes(nodeIndex)
        val newComponentsAllocation = actionSpace
          .actions(actionIndex)
          .map { case PairComponentDevice(component, device) =>
            val deviceID = device match {
              case MySelf()       => node.getId
              case EdgeServer(id) => id + applicationNodes.size
              case Cloud(id)      => id + applicationNodes.size + infrastructuralNodes.size
            }
            component.id -> deviceID
          }
          .toMap
        updateAllocation(node, newComponentsAllocation)
      }

  }

  private def scaleIds(matrix: Matrix, fromInfrastructural: Boolean = true): Matrix = {
    Matrix(matrix
      .data
      .map {
        case (from, to) =>
          val scaledFrom = if (fromInfrastructural) { from - applicationNodes.size } else { from }
          val scaledTo = to.map(_ - applicationNodes.size)
          (scaledFrom, scaledTo)
      })
  }

  protected def getNodeFeature(node: Node[T]): Vector = {
    if (!node.contains(new SimpleMolecule(Molecules.infrastructural)) && !node.contains(new SimpleMolecule(Molecules.cloud))) {
      val componentsAllocation = getAllocator(node).getComponentsAllocation
      val totalComponents = componentsAllocation.size
      val localComponents = componentsAllocation.count { case (_, where) => node.getId == where }
      val edgeServerComponents = componentsAllocation
        .count { case (_, where) => infrastructuralNodes.map(_.getId).contains(where) }
      val cloudComponents = componentsAllocation
        .count { case (_, where) => cloudNodes.map(_.getId).contains(where) }
      val edgeServerDeltaCost = getDeltaCost(infrastructuralNodes, node.getId)
      val cloudDeltaCost = getDeltaCost(cloudNodes, node.getId)
      val batteryLevel = BatteryEquippedDevice.getBatteryPercentage(node)

      val locations = componentsAllocation
        .values
        .map {
          case id if id == node.getId => -1.0
          case id => (id - applicationNodes.size).toDouble
        }
        .toSeq
      val latencies: Seq[Double] = componentsAllocation.map {
        case (componentId, where) if where == node.getId => 0.0
        case _ =>
          val density = node.getConcentration(new SimpleMolecule(Molecules.density)).asInstanceOf[Double]
          getLatency(density)
      }.toSeq

      val density = node.getConcentration(new SimpleMolecule(Molecules.density)).asInstanceOf[Double]

      val f = Seq(batteryLevel, edgeServerDeltaCost, cloudDeltaCost, localComponents, density) ++ locations //++ latencies
      Vector(f)
    } else {
      val cost = node.getConcentration(PayPerUseDevice.TOTAL_COST).asInstanceOf[Double]
      Vector(Seq(cost))
    }
  }

  private def getDeltaCost(nodes: Seq[Node[T]], mid: Int): Double =
    nodes
      .map(_.getId)
      .filter(remoteID => nodes.map(_.getId).contains(remoteID))
      .map(remoteID => getAlchemistActions(environment, remoteID, classOf[PayPerUseDevice[T, P]]))
      .map(_.head)
      .map(_.deltaCostPerDevice(mid))
      .sum

  private def getLatency(density: Double): Double = {
    density match {
      case d if d < 5.0   => 0.0
      case d if d < 15.0  => 0.2
      case _              => 2.0
    }
  }

  protected def getEdgeFeature(node: Node[T], neigh: Node[T]): Vector = {
    applicationNodes
      .map(_.getId)
      .contains(neigh.getId) match {
      case applicationNode if applicationNode =>
        val distance = environment.getPosition(node).distanceTo(environment.getPosition(neigh))
        Vector(Seq(distance))
      case _ =>
        val density = node.getConcentration(new SimpleMolecule(Molecules.density)).asInstanceOf[Double]
        val latency = getLatency(density)
        Vector(Seq(latency))
    }
  }

  private def toFeatures(data: Seq[Node[T]]): py.Dynamic = {
    val tensors = data.map(getNodeFeature).map(toTorchTensor(_, index = false))
    tensors.tail
      .foldLeft(tensors.head.unsqueeze(0))((acc, elem) => torch.cat((acc, elem.unsqueeze(0)), dim = 0))
  }

  private def createFeatureTensor(matrix: Matrix): py.Any = {
    toTorchTensor(
      matrix.allEdges.map(edge => getEdgeFeature(environment.getNodeByID(edge._1), environment.getNodeByID(edge._2))),
    )
  }

  private def filterNeighbors(neighbors: Seq[Int], nodes: Seq[Int]): Seq[Int] = {
    neighbors.filter(neigh => nodes.contains(neigh))
  }

  private def getEdgeIndexes(sourceNodes: Seq[Int], endNodes: Seq[Int]): Matrix = {
    val d = nodes
      .filter(n => sourceNodes.contains(n.getId))
      .map(n => (n.getId, getNeighbors(n)))
      .map { case (s, neighs) => (s, filterNeighbors(neighs, endNodes)) }
    Matrix(d)
  }

  private def getEdgeIndexesAll(sourceNodes: Seq[Int], endNodes: Seq[Int]): Matrix = {
    val d = sourceNodes
      .map(n => (n, endNodes))
    Matrix(d)
  }

  private def toTorchTensor(data: Tensor, index: Boolean = true): py.Dynamic = data match {
    case v: Vector =>
      if (index)
        torch.tensor(v.data.toPythonCopy, dtype = torch.int64)
      else
        torch.tensor(v.data.toPythonCopy)
    case m: Matrix =>
      val tensors = m.data
        .map { case (self, neighs) => List(List.fill(neighs.length)(self), neighs) }
        .map { m =>
          if (index)
            torch.tensor(m, dtype = torch.int64)
          else
            torch.tensor(m)
        }
      tensors.tail
        .foldLeft(torch.tensor(tensors.head))((acc, elem) => torch.cat((acc, elem), dim = 1))
  }

  private def toTorchTensor(data: Seq[Tensor]): py.Any = {
    val tensors = data.map(toTorchTensor(_, index = false))
    tensors.tail
      .foldLeft(tensors.head.unsqueeze(0))((acc, elem) => torch.cat((acc, elem.unsqueeze(0)), dim = 0))
  }

  protected def getComponents: Seq[Component] = {
    getAllocator(applicationNodes.head).getComponentsAllocation.keys
      .map(id => Component(id))
      .toSeq
  }

  protected def getAllocator(node: Node[T]): AllocatorProperty[T, P] = {
    node.getProperties.asScala
      .filter(_.isInstanceOf[AllocatorProperty[T, P]])
      .map(_.asInstanceOf[AllocatorProperty[T, P]])
      .head
  }

  protected def updateAllocation(node: Node[T], newAllocation: Map[String, Int]): Unit = {
    getAllocator(node)
      .setComponentsAllocation(newAllocation)

    val batteryModel = node.getReactions.asScala
      .flatMap(_.getActions.asScala)
      .find(_.isInstanceOf[BatteryEquippedDevice[T, P]])
      .map(_.asInstanceOf[BatteryEquippedDevice[T, P]])
      .getOrElse(throw new IllegalStateException("Battery action not found!"))

    batteryModel.updateComponentsExecution(newAllocation)

    val localComponents = newAllocation.values.count(_ == node.getId).toDouble
    val localComponentsPercentage = localComponents / components.size.toDouble
    node.setConcentration(
      new SimpleMolecule(Molecules.localComponentsPercentage),
      localComponentsPercentage.asInstanceOf[T],
    )
  }

  private def getNeighbors(n: Node[T]): Seq[Int] = {
    environment
      .getNeighborhood(n)
      .getNeighbors
      .stream()
      .map(neigh => neigh.getId)
      .toList
      .asScala
      .toList
  }

}
