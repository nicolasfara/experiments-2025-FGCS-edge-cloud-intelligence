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
    val adjacencyInfraToInfra = getEdgeIndexes(infrastructuralNodes.map(_.getId - applicationNodes.size), infrastructuralNodes.map(_.getId - applicationNodes.size))
    val adjacencyAppToInfra = getEdgeIndexes(applicationNodes.map(_.getId), infrastructuralNodes.map(_.getId - applicationNodes.size))
    val pyAdjacencyAppToApp = toTorchTensor(adjacencyAppToApp)
    val pyAdjacencyInfraToInfra = toTorchTensor(adjacencyInfraToInfra)
    val pyAdjacencyAppToInfra = toTorchTensor(adjacencyAppToInfra)
    val pyFeaturesApplication = toFeatures(applicationNodes)
    val pyFeaturesInfrastructural = toFeatures(infrastructuralNodes)
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
    tensors
      .tail
      .foldLeft(tensors.head.unsqueeze(0))((acc, elem) => torch.cat((acc, elem.unsqueeze(0)), dim = 0))
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

  private def toTorchTensor(data: Tensor, index: Boolean = true): py.Dynamic = data match {
    case v: Vector =>
      if (index)
        torch.tensor(v.data.toPythonCopy, dtype= torch.int64)
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
    tensors
      .tail
      .foldLeft(tensors.head.unsqueeze(0))((acc, elem) => torch.cat((acc, elem.unsqueeze(0)), dim = 0))
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
