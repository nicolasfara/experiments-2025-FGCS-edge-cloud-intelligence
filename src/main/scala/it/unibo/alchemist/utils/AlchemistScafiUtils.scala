package it.unibo.alchemist.utils

import it.unibo.alchemist.model.{Action, Environment, Node, Position}

import scala.jdk.CollectionConverters.IterableHasAsScala
import scala.reflect.ClassTag

object AlchemistScafiUtils {
  def getNeighborsWithProgram[T, P <: Position[P], PG <: Action[T] : ClassTag](node: Node[T])(implicit env: Environment[T, P]): List[Node[T]] = {
    (for {
      node <- env.getNeighborhood(node).asScala
      reaction <- node.getReactions.asScala
      action <- reaction.getActions.asScala
      if implicitly[ClassTag[PG]].runtimeClass.isInstance(action)
    } yield node).toList
  }
}
