package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.actions.AbstractLocalAction
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Action, AllocatorProperty, BatteryEquippedDevice, Environment, Node, Position, Reaction}
import org.apache.commons.math3.random.RandomGenerator

import scala.jdk.CollectionConverters.{CollectionHasAsScala, IteratorHasAsScala}

class RuleControllerModularizationRuntime[T, P <: Position[P]](
    environment: Environment[T, P],
    node: Node[T],
    random: RandomGenerator,
    componentsDag: Map[String, Set[String]],
    scenarioType: String,
) extends AbstractLocalAction[T](node) {
  private val infrastructuralMolecule = new SimpleMolecule("infrastructuralDevice")
  private val cloudDeviceMolecule = new SimpleMolecule("cloudDevice")
  private val components = componentsDag.keySet ++ componentsDag.values.flatten
  private lazy val allocator: AllocatorProperty[T, P] = node.getProperties.asScala
    .find(_.isInstanceOf[AllocatorProperty[T, P]])
    .map(_.asInstanceOf[AllocatorProperty[T, P]])
    .getOrElse(throw new IllegalStateException(s"`AllocatorProperty` not found for node ${node.getId}"))
  private lazy val batteryModel = node.getReactions.asScala
    .flatMap(_.getActions.asScala)
    .find(_.isInstanceOf[BatteryEquippedDevice[T, P]])
    .map(_.asInstanceOf[BatteryEquippedDevice[T, P]])
    .getOrElse(throw new IllegalStateException(s"`BatteryEquippedDevice` not found for node ${node.getId}"))
  private lazy val edgeSurrogate = environment
    .getNeighborhood(node)
    .iterator()
    .asScala
    .filter(_.contains(infrastructuralMolecule))
    .toList
  private lazy val cloudSurrogate = environment
    .getNodes
    .iterator()
    .asScala
    .filter(_.contains(cloudDeviceMolecule))
    .toList
  private lazy val candidateEdgeInfrastructural = edgeSurrogate(random.nextInt(edgeSurrogate.size))
  private lazy val candidateCloud = cloudSurrogate(random.nextInt(cloudSurrogate.size))

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = {
    scenarioType match {
      case "fulllocal"   => allocator.setComponentsAllocation(components.map(_ -> node.getId).toMap)
      case "fulloffloadedge" => allocator.setComponentsAllocation(components.map(_ -> candidateEdgeInfrastructural.getId).toMap)
      case "fulloffloadcloud" => allocator.setComponentsAllocation(components.map(_ -> candidateCloud.getId).toMap)
      case _              => throw new IllegalStateException(s"Unknown scenario type: $scenarioType")
    }
    batteryModel.updateComponentsExecution(allocator.getComponentsAllocation)
    val percentageOffloadedComponents = allocator.getComponentsAllocation.values.count(_ != node.getId) / components.size.toDouble
    node.setConcentration(RuleControllerModularizationRuntime.PERCENTAGE_OFFLOADED_COMPONENTS, percentageOffloadedComponents.asInstanceOf[T])
  }
}

private object RuleControllerModularizationRuntime {
  private val PERCENTAGE_OFFLOADED_COMPONENTS = new SimpleMolecule("percentageOffloadedComponents")
}
