package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Node, Position, TimeDistribution}
import it.unibo.alchemist.utils.Molecules
import scala.util.Random

class MoveNodes [T, P <: Position[P]](
  environment: Environment[T, P],
  distribution: TimeDistribution[T],
  seed: Int
) extends AbstractGlobalReaction[T, P](environment, distribution) {


  private lazy val applicationNodes: Seq[Node[T]] = nodes
    .filterNot(n => n.contains(new SimpleMolecule(Molecules.infrastructural)))
    .filterNot(n => n.contains(new SimpleMolecule(Molecules.cloud)))
    .sortBy(node => node.getId)

  private val random = new Random(seed)

  private val movingNodes = List(0, 1, 32)
  private val movementRange = -4 to -2
  private val deltaX = List.fill(movingNodes.size)(random.between(movementRange.start, movementRange.end))
  private val deltaY = List.fill(movingNodes.size)(random.between(movementRange.start, movementRange.end))

  override protected def executeBeforeUpdateDistribution(): Unit = {
    random
      .shuffle(applicationNodes)
      .take(movingNodes.size)
      .zipWithIndex
      .foreach { case (node, index) =>
        val newPosition = environment.getPosition(node).plus(Array(deltaX(index), deltaY(index)))
        environment.moveNodeToPosition(node, newPosition)
      }
  }
}
