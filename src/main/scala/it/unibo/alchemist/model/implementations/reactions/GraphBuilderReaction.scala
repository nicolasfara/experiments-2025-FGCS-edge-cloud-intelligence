package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{AllocatorProperty, DecayLayer, Environment, LearningLayer, Node, Position, TimeDistribution}

import scala.jdk.CollectionConverters.{CollectionHasAsScala, IteratorHasAsScala}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import it.unibo.alchemist.utils.Molecules
import learning.model.{ActionSpace, Cloud, Component, EdgeServer, MySelf, PairComponentDevice}

import scala.language.implicitConversions
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py
import me.shadaj.scalapy.readwrite.{Reader, Writer}

trait Tensor
case class Vector(data: Seq[Double]) extends Tensor
case class Matrix(data: Seq[(Int, Seq[Int])]) extends Tensor {
  def toCoordinateFormat: (Seq[Int], Seq[Int]) = {
    val startingEdges = data.map { case (self, neighs) => Seq.fill(neighs.length)(self) }
    val endEdges = data.flatMap { case (_, neighs) => neighs }
    (startingEdges.flatten, endEdges)
  }
  def allEdges: Seq[(Int, Int)] = data.flatMap { case (self, neighs) => neighs.map(neigh => (self, neigh)) }
}

abstract class GraphBuilderReaction[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
) extends AbstractGlobalReaction(environment, distribution) {

  protected lazy val cloudNodes: Seq[Node[T]] = nodes
    .filter(n => n.contains(Molecules.cloud))
    .sortBy(node => node.getId)

  protected lazy val infrastructuralNodes: Seq[Node[T]] = nodes
    .filter(n => n.contains(Molecules.infrastructural))
    .sortBy(node => node.getId)

  protected lazy val applicationNodes: Seq[Node[T]] = nodes
    .filterNot(n => n.contains(Molecules.infrastructural))
    .filterNot(n => n.contains(Molecules.cloud))
    .sortBy(node => node.getId)

  protected lazy val learner: py.Dynamic = environment
    .getLayer(new SimpleMolecule(Molecules.learner))
    .get()
    .asInstanceOf[LearningLayer[P]]
    .getValue(environment.makePosition(0, 0))

  private lazy val epsilon = environment
    .getLayer(new SimpleMolecule(Molecules.decay))
    .get()
    .asInstanceOf[DecayLayer[P]]
    .getValue(environment.makePosition(0, 0))

  private lazy val edgeServerSize = infrastructuralNodes.size

  private lazy val cloudSize = cloudNodes.size

  protected lazy val components: Seq[Component] = getComponents

  private var oldGraph: Option[py.Dynamic] = None

  private var oldActions: Option[py.Dynamic] = None

  protected var currentAllocation: Option[Map[String, Int]] = None

  private lazy val actionSpace = ActionSpace(components, edgeServerSize, cloudSize)

  private var executedToHetero = false

  private implicit def toMolecule(name: String): SimpleMolecule = new SimpleMolecule(name)

  private implicit def toPythonCollection[K: Reader: Writer](collection: Seq[K]): py.Any = collection.toPythonProxy

  private implicit def toPythonCollectionNested[K: Reader: Writer](collection: Seq[Seq[K]]): py.Any = collection.toPythonProxy

  override protected def executeBeforeUpdateDistribution(): Unit = handleGraph(createGraph())

  private def createGraph(): py.Dynamic = {

    val adjacencyAppToApp = getEdgeIndexes(applicationNodes.map(_.getId), applicationNodes.map(_.getId))
    val infraNodes = infrastructuralNodes.appendedAll(cloudNodes)
    val adjacencyInfraToInfra = getEdgeIndexes(infraNodes.map(_.getId - applicationNodes.size), infraNodes.map(_.getId - applicationNodes.size))
    val adjacencyAppToInfra = getEdgeIndexesAll(applicationNodes.map(_.getId), infraNodes.map(_.getId - applicationNodes.size))
    val pyAdjacencyAppToApp = toTorchTensor(adjacencyAppToApp)
    val pyAdjacencyInfraToInfra = toTorchTensor(adjacencyInfraToInfra)
    val pyAdjacencyAppToInfra = toTorchTensor(adjacencyAppToInfra)
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

  protected def getNodeFeature(node: Node[T]): Vector

  protected def getEdgeFeature(node: Node[T], neigh: Node[T]): Vector

  protected def handleGraph(observation: py.Dynamic): Unit = {

    if (!executedToHetero) {
      learner.toHetero(observation)
      executedToHetero = true
    }

    val actions = learner.select_action(observation, epsilon)
//    println(environment.getSimulation.getTime.toDouble)
//    println(s"[DEBUG] $actions")
//    var actions = torch.full((1, 100), 0).flatten()
//    val time = environment.getSimulation.getTime.toDouble
//    if (time >= 15 && time < 30 ){
//      actions = torch.full((1, 100), 34).flatten()
//    } else {
//      actions = torch.full((1, 100), 0).flatten()
//    }
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

    (oldGraph, oldActions) match {
      case (Some(previousObs), Some(previousActions)) =>
        val rewards = computeRewards(previousObs, observation)
        learner.add_experience(previousObs, previousActions, rewards, observation)
        val loss = learner.train_step_dqn(batch_size = 32, gamma = 0.99, seed = getSeed).as[Double]
        environment.getNodes.iterator().asScala.foreach(_.setConcentration(new SimpleMolecule("loss"), loss.asInstanceOf[T]))
      case _ =>
    }
    oldGraph = Some(observation)
    oldActions = Some(actions)
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

  protected def updateAllocation(node: Node[T], newAllocation: Map[String, Int]): Unit

  protected def computeRewards(obs: py.Dynamic, nextObs: py.Dynamic): py.Dynamic

  protected def getSeed: Int

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
