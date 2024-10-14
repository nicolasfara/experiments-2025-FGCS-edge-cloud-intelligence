package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Node, Position, TimeDistribution}

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
case class Matrix(data: Seq[(Int, Seq[Int])]) extends Tensor

abstract class GraphBuilderReaction[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
) extends AbstractGlobalReaction(environment, distribution) {

  private implicit def toMolecule(name: String): SimpleMolecule = new SimpleMolecule(name)
  private implicit def toPythonCollection[K: Reader: Writer](collection: Seq[K]): py.Any = collection.toPythonProxy
  private implicit def toPythonCollectionNested[K: Reader: Writer](collection: Seq[Seq[K]]): py.Any = collection.toPythonProxy

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val infrastructuralNodes = nodes
      .filter(n => n.contains(Molecules.infrastructural))
    // .map(n => n.getId)
    val applicationNodes = nodes
      .filterNot(n => n.contains(Molecules.infrastructural))
    // .map(n => n.getId)
    val app2app = toTorchTensor(getEdgeIndexes(applicationNodes.map(_.getId), applicationNodes.map(_.getId)))
    val infr2infr = toTorchTensor(getEdgeIndexes(infrastructuralNodes.map(_.getId), infrastructuralNodes.map(_.getId)))
    val app2infr = toTorchTensor(getEdgeIndexes(applicationNodes.map(_.getId()), infrastructuralNodes.map(_.getId)))
    val featuresInfrastructural = infrastructuralNodes.map(getFeature).map(toTorchTensor)
    val featuresApplication = applicationNodes.map(getFeature).map(toTorchTensor)
  }

  protected def getFeature(node: Node[T]): Vector

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
    case v: Vector => torch.Tensor(v.data, dtype = torch.float64)
    case m: Matrix =>
      val tensors = m.data
        .map { case (self, neighs) => List(List.fill(neighs.length)(self), neighs) }
        .map { m => torch.tensor(m, dtype = torch.long) }
      tensors.tail
        .foldLeft(torch.tensor(tensors.head, dtype = torch.long))((elem, acc) => torch.cat((acc, elem), dim = 1))
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
