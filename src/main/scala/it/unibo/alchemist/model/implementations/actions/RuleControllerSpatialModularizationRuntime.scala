package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.actions.AbstractLocalAction
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model._
import org.apache.commons.math3.random.RandomGenerator

import java.lang.System.Logger.Level
import scala.jdk.CollectionConverters.{CollectionHasAsScala, IteratorHasAsScala}

class RuleControllerSpatialModularizationRuntime[T, P <: Position[P]](
    environment: Environment[T, P],
    node: Node[T],
    random: RandomGenerator,
    componentsDag: Map[String, Set[String]],
    neighborsThreshold: Int,
    neighborsLevel: Int,
) extends AbstractLocalAction[T](node) {
  private val infrastructuralMolecule = new SimpleMolecule("infrastructuralDevice")
  private val components = componentsDag.keySet ++ componentsDag.values.flatten
  private lazy val allocator: AllocatorProperty[T, P] = node.getProperties.asScala
    .find(_.isInstanceOf[AllocatorProperty[T, P]])
    .map(_.asInstanceOf[AllocatorProperty[T, P]])
    .getOrElse(throw new IllegalStateException(s"`AllocatorProperty` not found for node ${node.getId}"))
  private lazy val surrogate = environment
    .getNeighborhood(node)
    .iterator()
    .asScala
    .filter(_.contains(infrastructuralMolecule))
    .toList
  private lazy val candidateInfrastructural = surrogate(random.nextInt(surrogate.size))
  private lazy val neighborsForNode = getNeighborsWithLevel(node, neighborsLevel)

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = {
    if (neighborsForNode.size >= neighborsThreshold) {
      allocator.setComponentsAllocation(components.map(_ -> candidateInfrastructural.getId).toMap)
    } else {
      allocator.setComponentsAllocation(components.map(_ -> node.getId).toMap)
    }
    val percentageOffloadedComponents = allocator.getComponentsAllocation.values.count(_ != node.getId) / components.size.toDouble
    node.setConcentration(RuleControllerSpatialModularizationRuntime.PERCENTAGE_OFFLOADED_COMPONENTS, percentageOffloadedComponents.asInstanceOf[T])
  }

  private def getNeighborsWithLevel(node: Node[T], level: Int): Set[Node[T]] = {
    level match {
      case 0 => Set(node)
      case _ =>
        environment
          .getNeighborhood(node)
          .getNeighbors
          .iterator()
          .asScala
          .filterNot(_.contains(infrastructuralMolecule))
          .flatMap { neighbor => getNeighborsWithLevel(neighbor, level - 1) }
          .toSet
    }
  }
}

private object RuleControllerSpatialModularizationRuntime {
  private val PERCENTAGE_OFFLOADED_COMPONENTS = new SimpleMolecule("percentageOffloadedComponents")
}
