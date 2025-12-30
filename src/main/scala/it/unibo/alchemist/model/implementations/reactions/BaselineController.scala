package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{AllocatorProperty, BatteryEquippedDevice, Environment, Node, Position, TimeDistribution}
import it.unibo.alchemist.utils.Molecules
import learning.model.Component

import scala.jdk.CollectionConverters.CollectionHasAsScala
import scala.language.implicitConversions

sealed trait Experiment
case object AlwaysLocal extends Experiment
case object AlwaysEdgeServer extends Experiment
case object AlwaysCloud extends Experiment
case object Random extends Experiment

object ExperimentsSpace {

  private val experiments = List(AlwaysLocal, AlwaysEdgeServer, AlwaysCloud, Random)

  def fromIndex(index: Int): Experiment = experiments(index)

}

class BaselineController [T, P <: Position[P]](
  environment: Environment[T, P],
  distribution: TimeDistribution[T],
  experimentName: Int,
  seed: Int
) extends AbstractGlobalReaction(environment, distribution) {

  private val random = new scala.util.Random(seed)

  private implicit def toMolecule(name: String): SimpleMolecule = new SimpleMolecule(name)

  private val experiment = ExperimentsSpace.fromIndex(experimentName)

  protected lazy val cloudNodes: Seq[Node[T]] = nodes
    .filter(n => n.contains(Molecules.cloud))
    .sortBy(node => node.getId)

  protected lazy val infrastructuralNodes: Seq[Node[T]] = nodes
    .filter(n => n.contains(Molecules.infrastructural))
    .sortBy(node => node.getId)

  protected lazy val applicationNodes: Seq[Node[T]] = nodes
    .filterNot(n => n.contains(Molecules.infrastructural))
    .filterNot(n => n.contains(Molecules.cloud))
    .sortBy(node => node.getId)

  private lazy val components = getComponents

  private var executed = false

  override protected def executeBeforeUpdateDistribution(): Unit = {

    if(executed){
      applicationNodes.foreach { node =>
        val newComponentsAllocation = components.map(_.id -> offloadTo(node)).toMap
        updateAllocation(node, newComponentsAllocation)
      }
    }
    executed = true

  }

  private def offloadTo(node: Node[T]): Int = experiment match {
    case AlwaysLocal => node.getId
    case AlwaysEdgeServer => infrastructuralNodes.head.getId
    case AlwaysCloud => cloudNodes.head.getId
    case Random => random.shuffle(List(node.getId, infrastructuralNodes.head.getId,cloudNodes.head.getId )).head
  }

  protected def updateAllocation(node: Node[T], newAllocation: Map[String, Int]): Unit = {
    getAllocator(node)
      .setComponentsAllocation(newAllocation)

    val batteryModel = node.getReactions.asScala
      .flatMap(_.getActions.asScala)
      .find(_.isInstanceOf[BatteryEquippedDevice[T, P]])
      .map(_.asInstanceOf[BatteryEquippedDevice[T, P]])
      .getOrElse(throw new IllegalStateException("Battery action not found!"))

    batteryModel.updateComponentsExecution(newAllocation)

    val localComponents = newAllocation.values.count(_ == node.getId).toDouble
    val localComponentsPercentage = localComponents / components.size.toDouble
    node.setConcentration(
      new SimpleMolecule(Molecules.localComponentsPercentage),
      localComponentsPercentage.asInstanceOf[T],
    )
  }

  protected def getComponents: Seq[Component] = {
    if(applicationNodes.nonEmpty){
      getAllocator(applicationNodes.head).getComponentsAllocation.keys
        .map(id => Component(id))
        .toSeq
    } else {
      Seq()
    }

  }

  protected def getAllocator(node: Node[T]): AllocatorProperty[T, P] = {
    node.getProperties.asScala
      .filter(_.isInstanceOf[AllocatorProperty[T, P]])
      .map(_.asInstanceOf[AllocatorProperty[T, P]])
      .head
  }
}
