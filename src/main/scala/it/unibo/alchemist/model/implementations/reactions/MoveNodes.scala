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

  private val movingNodes = 15
  private val movementRange = -2 to 3
  private val deltaX = List.fill(movingNodes)(random.between(movementRange.start, movementRange.end))
  private val deltaY = List.fill(movingNodes)(random.between(movementRange.start, movementRange.end))

  override protected def executeBeforeUpdateDistribution(): Unit = {
    println("----------------------- [DEBUG] Moving nodes -----------------------")
    random
      .shuffle(applicationNodes)
      .take(movingNodes)
      .zipWithIndex
      .foreach { case (node, index) =>
        println("DIOCANEEEEE")
        val newPosition = environment.getPosition(node).plus(Array(deltaX(index), deltaY(index)))
        environment.moveNodeToPosition(node, newPosition)
      }
  }
}
