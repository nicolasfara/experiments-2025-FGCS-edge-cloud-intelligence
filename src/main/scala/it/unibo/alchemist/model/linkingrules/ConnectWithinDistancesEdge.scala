package it.unibo.alchemist.model.linkingrules

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.neighborhoods.Neighborhoods
import it.unibo.alchemist.model.{Environment, LinkingRule, Neighborhood, Node, Position}

import scala.jdk.CollectionConverters.{CollectionHasAsScala, IterableHasAsJava, IteratorHasAsScala}

class ConnectWithinDistancesEdge[T, P <: Position[P]](private val radius: Double) extends LinkingRule[T, P] {
  private val infrastructuralMolecule = new SimpleMolecule("infrastructuralDevice")

  override def computeNeighborhood(node: Node[T], environment: Environment[T, P]): Neighborhood[T] = {
    if (node.contains(infrastructuralMolecule)) {
      Neighborhoods.make(environment, node, environment.getNodes)
    } else {
      val neighbors = environment.getNodesWithinRange(node, radius).iterator().asScala.toList
      val surrogates = environment.getNodes.asScala.filter(_.contains(infrastructuralMolecule)).toList
      val nodes = neighbors ++ surrogates
      Neighborhoods.make(environment, node, nodes.asJava)
    }
  }

  override def isLocallyConsistent: Boolean = false
}
