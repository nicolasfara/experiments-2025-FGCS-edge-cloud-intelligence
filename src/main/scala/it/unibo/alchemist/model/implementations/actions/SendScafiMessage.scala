package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.{Action, Context, Environment, Node, Position, Reaction}
import it.unibo.alchemist.model.actions.AbstractAction
import it.unibo.alchemist.model.implementations.nodes.ScafiDevice

sealed abstract class SendScafiMessage[T, P <: Position[P]](
    device: ScafiDevice[T],
    val program: RunScafiProgram[T, P]
) extends AbstractAction[T](device.getNode())

final class SendApplicationScafiMessage[T, P <: Position[P]](
    environment: Environment[T, P],
    device: ScafiDevice[T],
    reaction: Reaction[T],
    override val program: RunScafiProgram[T, P]
) extends SendScafiMessage[T, P](device, program) {

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = ???

  override def getContext: Context = ???
}

final class SendSurrogateScafiMessage[T, P <: Position[P]](
    environment: Environment[T, P],
    device: ScafiDevice[T],
    reaction: Reaction[T],
    override val program: RunScafiProgram[T, P]
) extends SendScafiMessage[T, P](device, program) {

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = ???

  override def getContext: Context = ???
}
