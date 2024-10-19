package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, LearningLayer, Node, Position, TimeDistribution}

import scala.jdk.CollectionConverters.CollectionHasAsScala
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import it.unibo.alchemist.utils.Molecules

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

  private implicit def toMolecule(name: String): SimpleMolecule = new SimpleMolecule(name)
  private implicit def toPythonCollection[K: Reader: Writer](collection: Seq[K]): py.Any = collection.toPythonProxy
  private implicit def toPythonCollectionNested[K: Reader: Writer](collection: Seq[Seq[K]]): py.Any = collection.toPythonProxy

  override protected def executeBeforeUpdateDistribution(): Unit = handleGraph(createGraph())

  protected lazy val infrastructuralNodes: Seq[Node[T]] = nodes
    .filter(n => n.contains(Molecules.infrastructural))
    .sortBy(node => node.getId)
  protected lazy val applicationNodes: Seq[Node[T]] = nodes
    .filterNot(n => n.contains(Molecules.infrastructural))
    .sortBy(node => node.getId)

  protected lazy val learner: py.Dynamic = environment
    .getLayer(new SimpleMolecule(Molecules.learner))
    .get()
    .asInstanceOf[LearningLayer[P]]
    .getValue(environment.makePosition(0, 0))

  private def createGraph(): py.Dynamic = {
    val adjacencyAppToApp = getEdgeIndexes(applicationNodes.map(_.getId), applicationNodes.map(_.getId))
    val adjacencyInfraToInfra = getEdgeIndexes(infrastructuralNodes.map(_.getId), infrastructuralNodes.map(_.getId))
    val adjacencyAppToInfra = getEdgeIndexes(applicationNodes.map(_.getId), infrastructuralNodes.map(_.getId))
    val pyAdjacencyAppToApp = toTorchTensor(adjacencyAppToApp)
    val pyAdjacencyInfraToInfra = toTorchTensor(adjacencyInfraToInfra)
    val pyAdjacencyAppToInfra = toTorchTensor(adjacencyAppToInfra)
    val featuresApplication = applicationNodes.map(getNodeFeature).map(toTorchTensor)
    val featuresInfrastructural = infrastructuralNodes.map(getNodeFeature).map(toTorchTensor)
    val featureAppToApp = createFeatureTensor(adjacencyAppToApp)
    val featureInfraToInfra = createFeatureTensor(adjacencyInfraToInfra)
    val featureAppToInfra = createFeatureTensor(adjacencyAppToInfra)
    rlUtils.create_graph(
      featuresApplication,
      featuresInfrastructural,
      pyAdjacencyAppToApp,
      pyAdjacencyInfraToInfra,
      pyAdjacencyAppToInfra,
      featureAppToApp,
      featureInfraToInfra,
      featureAppToInfra,
    )
  }

  private def createFeatureTensor(matrix: Matrix): py.Any = {
    toTorchTensor(
      matrix.allEdges.map(edge => getEdgeFeature(environment.getNodeByID(edge._1), environment.getNodeByID(edge._2))),
    )
  }

  protected def getNodeFeature(node: Node[T]): Vector
  protected def getEdgeFeature(node: Node[T], neigh: Node[T]): Vector
  protected def handleGraph(graph: py.Dynamic): Unit
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

  private def toTorchTensor(data: Tensor): py.Dynamic = data match {
    case v: Vector => torch.tensor(v.data.toPythonCopy, dtype = torch.float64)
    case m: Matrix =>
      val tensors = m.data
        .map { case (self, neighs) => List(List.fill(neighs.length)(self), neighs) }
        .map { m => torch.tensor(m, dtype = torch.long) }
      tensors.tail
        .foldLeft(torch.tensor(tensors.head, dtype = torch.long))((elem, acc) => torch.cat((acc, elem), dim = 1))
  }

  private def toTorchTensor(data: Seq[Tensor]): py.Any = data.map(toTorchTensor)
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
