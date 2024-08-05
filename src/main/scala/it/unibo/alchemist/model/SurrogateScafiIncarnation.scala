package it.unibo.alchemist.model
import it.unibo.alchemist.model.implementations.actions.{RunApplicationScafiProgram, RunScafiProgram, RunSurrogateScafiProgram}
import it.unibo.alchemist.model.implementations.conditions.ScafiComputationalRoundComplete
import it.unibo.alchemist.model.implementations.nodes.ScafiDevice
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.nodes.GenericNode
import it.unibo.alchemist.model.reactions.{ChemicalReaction, Event}
import it.unibo.alchemist.model.timedistributions.{DiracComb, ExponentialTime}
import it.unibo.alchemist.model.times.DoubleTime
import it.unibo.alchemist.scala.ScalaInterpreter
import org.apache.commons.math3.random.RandomGenerator

import java.util.Objects
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.{BufferHasAsJava, CollectionHasAsScala}
import scala.reflect.ClassTag

class SurrogateScafiIncarnation[T, P <: Position[P]] extends Incarnation[T, P] {
  private[this] def notNull[V](value: V, name: String = "Object"): V =
    Objects.requireNonNull(value, s"$name must not be null")

  private[this] def toDouble(value: Any): Double = value match {
    case x: String  => java.lang.Double.parseDouble(x)
    case x: Boolean => if (x) 1 else 0
    case x: Long    => x.toDouble
    case x: Double  => x
    case x: Int => x
    case x: Float => x
    case x:  Byte => x
    case x: Short  => x
    case _          => Double.NaN
  }

  override def getProperty(node: Node[T], molecule: Molecule, propertyName: String): Double = {
    val target = node.getConcentration(molecule)
    Option(propertyName).filter(_.trim.nonEmpty) match {
      case Some(prop) => toDouble(ScalaInterpreter(s"val value = $target; $prop"))
      case None       => toDouble(target)
    }
  }

  override def createMolecule(value: String): Molecule = new SimpleMolecule(notNull(value, "simple molecule name"))

  override def createConcentration(data: Any): T = {
    val dataString = data.toString
    val doCacheValue = !dataString.startsWith("_");
    CachedInterpreter[AnyRef](if (doCacheValue) dataString else dataString.tail, doCacheValue).asInstanceOf[T]
  }

  override def createConcentration(): T = null.asInstanceOf[T]

  override def createNode(
     randomGenerator: RandomGenerator,
     environment: Environment[T, P],
     parameter: Any
  ): Node[T] = {
    val scafiNode = new GenericNode[T](this, environment)
    scafiNode.addProperty(new ScafiDevice(scafiNode))
    scafiNode
  }

  override def createTimeDistribution(
    randomGenerator: RandomGenerator,
    environment: Environment[T, P],
    node: Node[T],
    parameters: Any
  ): TimeDistribution[T] = {
    Option(parameters) match {
      case None => new ExponentialTime[T](Double.PositiveInfinity, randomGenerator)
      case Some(param) =>
        val frequency = toDouble(param)
        if (frequency.isNaN) {
          throw new IllegalArgumentException(s"$param is not a valid number, the time distribution could not be created.")
        }
        new DiracComb(new DoubleTime(randomGenerator.nextDouble() / frequency), frequency)
    }
  }

  override def createReaction(
     randomGenerator: RandomGenerator,
     environment: Environment[T, P],
     node: Node[T],
     timeDistribution: TimeDistribution[T],
     parameters: Any
  ): Reaction[T] = {
    val parameterString = Option(parameters).map(_.toString)
    val isSend = parameterString.exists(parameter => parameter.equalsIgnoreCase("send") || parameter.equalsIgnoreCase("sendSurrogate"))

    val result: Reaction[T] = if (isSend) {
      new ChemicalReaction[T](
        Objects.requireNonNull(node),
        Objects.requireNonNull(timeDistribution)
      )
    } else {
      new Event[T](node, timeDistribution)
    }

    parameterString.foreach { param =>
      result.setActions(
        ListBuffer[Action[T]](createAction(randomGenerator, environment, node, timeDistribution, result, param)).asJava
      )
    }
    if (isSend) {
      result.setConditions(
        ListBuffer[Condition[T]](createCondition(randomGenerator, environment, node, timeDistribution, result, null)).asJava
      )
    }
    result
  }

  override def createCondition(
    randomGenerator: RandomGenerator,
    environment: Environment[T, P],
    node: Node[T],
    time: TimeDistribution[T],
    actionable: Actionable[T],
    additionalParameters: Any
  ): Condition[T] = SurrogateScafiIncarnation.runInScafiDeviceContext[T, Condition[T]](
    node,
    message = s"The node must have a ${classOf[ScafiDevice[_]].getSimpleName} property",
    device => {
      val isSurrogate = node.getReactions.asScala.flatMap(_.getActions.asScala).exists(_.isInstanceOf[RunSurrogateScafiProgram[_, _]])
      val programClazz = if (isSurrogate) classOf[RunSurrogateScafiProgram[T, P]] else classOf[RunApplicationScafiProgram[T, P]]
      val alreadyDone = SurrogateScafiIncarnation.allConditionsFor(node, programClazz)
        .map(_.asInstanceOf[ScafiComputationalRoundComplete[T, P, _]])
        .map(_.program.asInstanceOf[RunScafiProgram[T, P]])
      val allScafiPrograms = SurrogateScafiIncarnation.allScafiProgramsForType[T, P](node, programClazz)
      if (allScafiPrograms.isEmpty) {
        throw new IllegalStateException(s"There is no program requiring a ${programClazz.getSimpleName} condition")
      }
      if (allScafiPrograms.size > 1) {
        throw new IllegalStateException(s"There are too many programs requiring a ${programClazz.getSimpleName} condition: $allScafiPrograms")
      }
      val programsToComplete = allScafiPrograms.filterNot(t => alreadyDone.exists(e => e == t))
      return new ScafiComputationalRoundComplete(device, programsToComplete.head.asInstanceOf[RunScafiProgram[T, P]], programClazz)
    }
  )

  override def createAction(
    randomGenerator: RandomGenerator,
    environment: Environment[T, P],
    node: Node[T],
    time: TimeDistribution[T],
    actionable: Actionable[T],
    additionalParameters: Any
  ): Action[T] = ???
}

object SurrogateScafiIncarnation {
  def runInScafiDeviceContext[T, A](node: Node[T], message: String, body: ScafiDevice[T] => A): A = {
    if (!isScafiNode(node)) {
      throw new IllegalArgumentException(message)
    }
    body(node.asProperty(classOf[ScafiDevice[T]]))
  }

  def runOnlyOnScafiDevice[T, A](node: Node[T], message: String)(body: => A): A =
    runInScafiDeviceContext(node, message, (_: ScafiDevice[T]) => body)

  def isScafiNode[T](node: Node[T]): Boolean = node.asPropertyOrNull[ScafiDevice[T]](classOf[ScafiDevice[T]]) != null

  def allActions[T, P <: Position[P], C](node: Node[T], klass: Class[C]): Iterable[C] =
    for {
      reaction: Reaction[T] <- node.getReactions.asScala
      action: Action[T] <- reaction.getActions.asScala if klass.isInstance(action)
    } yield action.asInstanceOf[C]

  def allScafiProgramsForType[T, P <: Position[P], ProgType <: RunScafiProgram[T, P] : ClassTag](node: Node[T]) =
    allActions[T, P, ProgType](node, implicitly[ClassTag[ProgType]].runtimeClass.asInstanceOf[Class[ProgType]])

  def allScafiProgramsForType[T, P <: Position[P]](node: Node[T], clazz: Class[_ <: RunScafiProgram[T, P]]) =
    for {
      reaction: Reaction[T] <- node.getReactions.asScala
      action: Action[T] <- reaction.getActions.asScala if clazz.isInstance(action)
    } yield action

//
//  def allScafiProgramsFor[T, P <: Position[P]](node: Node[T]) =
//    allActions[T, P, RunScafiProgram[T, P]](node, classOf[RunScafiProgram[T, P]])
//
//  def allSurrogateScafiProgramsFor[T, P <: Position[P]](node: Node[T]) =
//    allActions[T, P, RunSurrogateScafiProgram[T, P]](node, classOf[RunSurrogateScafiProgram[T, P]])
//
  def allConditionsFor[T](node: Node[T], conditionClass: Class[_]): Iterable[Condition[T]] =
    for {
      reaction <- node.getReactions.asScala
      condition <- reaction.getConditions.asScala if conditionClass.isInstance(condition)
    } yield condition
}
