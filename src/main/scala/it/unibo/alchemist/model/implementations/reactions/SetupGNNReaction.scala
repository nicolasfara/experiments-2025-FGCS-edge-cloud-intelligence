package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, LearningLayer, Node, Position, TimeDistribution}
import it.unibo.alchemist.utils.Molecules
import me.shadaj.scalapy.py

/**
 * Global Reaction that setups the GNN, it must be executed only one time at the beginning of the simulation,
 * otherwise the GNN will not process heterogeneous data
 */
class SetupGNNReaction [T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
  ) extends GraphBuilderReaction[T, P](environment, distribution) {

  lazy val learner: py.Dynamic = environment
    .getLayer(new SimpleMolecule(Molecules.learner))
    .asInstanceOf[LearningLayer[P]]
    .getValue(environment.makePosition(0, 0))

  override protected def getNodeFeature(node: Node[T]): Vector = {
    val position = environment.getPosition(node)
    Vector(Seq(position.getCoordinate(0), position.getCoordinate(0)))
  }

  override protected def getEdgeFeature(node: Node[T], neigh: Node[T]): Vector = {
    val distance = environment.getPosition(node).distanceTo(environment.getPosition(neigh))
    Vector(Seq(distance))
  }

  override protected def handleGraph(graph: py.Any): Unit = {
    learner.toHetero(graph)
  }
}
