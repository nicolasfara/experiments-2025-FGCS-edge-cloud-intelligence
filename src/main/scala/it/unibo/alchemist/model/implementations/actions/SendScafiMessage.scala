package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.{Action, Context, Environment, Node, Position, Reaction, SurrogateScafiIncarnation, Target}
import it.unibo.alchemist.model.actions.AbstractAction
import it.unibo.alchemist.model.implementations.nodes.ScafiDevice
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.ID
import it.unibo.alchemist.utils.AlchemistScafiUtils.getNeighborsWithProgram
import it.unibo.alchemist.utils.ScalaJavaInterop.EnvironmentOps

import scala.jdk.CollectionConverters.{CollectionHasAsScala, IterableHasAsScala, IteratorHasAsScala}
import scala.reflect.ClassTag

sealed abstract class SendScafiMessage[T, P <: Position[P]](
    device: ScafiDevice[T],
    val program: RunScafiProgram[T, P]
) extends AbstractAction[T](device.getNode())

final class SendApplicationScafiMessage[T, P <: Position[P]](
    environment: Environment[T, P],
    device: ScafiDevice[T],
    reaction: Reaction[T],
    override val program: RunApplicationScafiProgram[T, P]
) extends SendScafiMessage[T, P](device, program) {
  assert(reaction != null, "Reaction cannot be null")
  assert(program != null, "Program cannot be null")

//  private val targetMolecule = new SimpleMolecule("Target")
  private var messagesExchanged = 0
  private implicit val env: Environment[T, P] = environment

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = {
    // Send to application neighbors
    val applicationNeighbors = getNeighborsWithProgram[T, P, RunApplicationScafiProgram[T, P]](getNode)
    applicationNeighbors.foreach(sendToNode)
    messagesExchanged = applicationNeighbors.size
    getNode.setConcentration(new SimpleMolecule("MessagesExchanged"), messagesExchanged.asInstanceOf[T])

    // Get programs to input the computed value
    for {
      componentsToInput <- program.programDag.get(program.programNameMolecule.getName)
      component <- componentsToInput
      neighbor <- applicationNeighbors :+ getNode // Add self node since is another program instance not having the input field
    } sendToInputOfComponent(neighbor, component)
    program.prepareForComputationalCycle()
  }

  override def getContext: Context = Context.NEIGHBORHOOD

  private def sendToInputOfComponent(node: Node[T], component: String): Unit = {
    val inputProgram = SurrogateScafiIncarnation
      .allScafiProgramsForType(node, classOf[RunApplicationScafiProgram[T, P]])
      .map(_.asInstanceOf[RunApplicationScafiProgram[T, P]])
      .find(_.asMolecule.getName == component)
      .getOrElse(throw new IllegalStateException(s"Program $component not found on node ${node.getId}"))
    val (path, optionalValue) = program.generateComponentOutputField()
    optionalValue match {
      case Some(value) => inputProgram.feedInputFromNode(device.getNode().getId, path -> value)
      case _           => println(s"No data available to feed input of $component on node ${node.getId} from ${device.getNode.getId}")
    }
  }

  private def sendToNode(node: Node[T]): Unit = {
    val programNode = SurrogateScafiIncarnation
      .allScafiProgramsForType(node, classOf[RunApplicationScafiProgram[T, P]])
      .map(_.asInstanceOf[RunApplicationScafiProgram[T, P]])
      .find(_.programNameMolecule == program.programNameMolecule)
      .getOrElse(throw new IllegalStateException(s"Program ${program.programNameMolecule} not found on node ${node.getId}"))
    program.getExport(device.getNode().getId) match {
      case Some(toSend) => programNode.sendExport(device.getNode().getId, toSend)
      case _            => println(s"No data available to send for ${device.getNode.getId} to ${node.getId}, maybe the program has been forwarded")
    }
  }

  def getNeighborsProgramByType[T, P <: Position[P], PG <: Action[T]: ClassTag](node: Node[T])(implicit env: Environment[T, P]): List[PG] = {
    (for {
      node <- env.getNeighborhood(node).asScala
      reaction <- node.getReactions.asScala
      action <- reaction.getActions.asScala
      if implicitly[ClassTag[PG]].runtimeClass.isInstance(action)
    } yield action.asInstanceOf[PG]).toList
  }

//  private def getNeighborsWithTarget(target: Target): List[Node[T]] = {
//    environment
//      .getNeighborhood(getNode)
//      .getNeighbors
//      .iterator()
//      .asScala
//      .filter(_.getConcentration(targetMolecule) == target.asInstanceOf[T])
//      .toList
//  }
}

final class SendSurrogateScafiMessage[T, P <: Position[P]](
    environment: Environment[T, P],
    device: ScafiDevice[T],
    reaction: Reaction[T],
    override val program: RunSurrogateScafiProgram[T, P]
) extends SendScafiMessage[T, P](device, program) {
  assert(reaction != null, "Reaction cannot be null")
  assert(program != null, "Program cannot be null")

  private var messageExchanged = 0
  override def getContext: Context = Context.NEIGHBORHOOD

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = {
    /*
     * Rationale: I need to send back the computed result to the original node since the other `send` will propagate
     * the the neighbors not managed by this device.
     * This is the case when the physical neighborhood is not fully offloaded to this device.
     */
    program.isSurrogateFor.foreach(nodeId => {
      for {
        toSend <- program.getComputedResultFor(nodeId)
        localProgram <- getLocalProgramForNode(nodeId)
      } {
        localProgram.sendExport(nodeId, toSend)
        localProgram.setResultWhenOffloaded(toSend.exportData.root())
      }
    })
    messageExchanged = program.isSurrogateFor.size
    getNode.setConcentration(new SimpleMolecule("MessagesExchanged"), messageExchanged.asInstanceOf[T])
    program.prepareForComputationalCycle()
  }

  private def getLocalProgramForNode(nodeId: ID): Option[RunApplicationScafiProgram[T, P]] = {
    val localPrograms = for {
      node <- environment.getNodesAsScala
      reactions <- node.getReactions.asScala
      action <- reactions.getActions.asScala
    } yield action match {
      case prog: RunApplicationScafiProgram[T, P] => if (program.asMolecule == prog.programNameMolecule) prog else null
      case _                                      => null
    }
    localPrograms.filter(_ != null).find(action => action.nodeManager.node.getId == nodeId)
  }
}
