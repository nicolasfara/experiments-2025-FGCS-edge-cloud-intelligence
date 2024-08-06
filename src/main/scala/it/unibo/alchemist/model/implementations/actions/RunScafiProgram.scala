package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.{Action, Environment, Molecule, Node, Position, Reaction}
import it.unibo.alchemist.model.actions.AbstractLocalAction
import it.unibo.alchemist.model.molecules.SimpleMolecule
import org.apache.commons.math3.random.RandomGenerator

sealed abstract class RunScafiProgram[T, P <: Position[P]](node: Node[T]) extends AbstractLocalAction[T](node) {
  def asMolecule: Molecule = new SimpleMolecule(getClass.getSimpleName)
  def isComputationalCycleComplete: Boolean
}

final class RunApplicationScafiProgram[T, P <: Position[P]](
    environment: Environment[T, P],
    node: Node[T],
    reaction: Reaction[T],
    randomGenerator: RandomGenerator,
    programName: String,
    retentionTime: Double,
    programDagMapping: Map[String, List[String]] = Map.empty
) extends RunScafiProgram[T, P](node) {
  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = ???

  override def isComputationalCycleComplete: Boolean = ???
}

final class RunSurrogateScafiProgram[T, P <: Position[P]](
    environment: Environment[T, P],
    node: Node[T],
    reaction: Reaction[T],
    randomGenerator: RandomGenerator,
    programName: String,
    retentionTime: Double,
    programDagMapping: Map[String, List[String]] = Map.empty
) extends RunScafiProgram[T, P](node) {
  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = ???

  override def isComputationalCycleComplete: Boolean = ???
}
