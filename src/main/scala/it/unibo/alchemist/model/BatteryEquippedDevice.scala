package it.unibo.alchemist.model

import it.unibo.alchemist.model.AllocatorProperty.ComponentId
import it.unibo.alchemist.model.actions.AbstractLocalAction

class BatteryEquippedDevice[T, P <: Position[P]](
    private val environment: Environment[T, P],
    private val node: Node[T],
    val batteryCapacity: Double, // mAh
    val energyPerInstruction: Double, // nJ
    val componentsInstructions: Map[ComponentId, Long],
    val batteryVoltage: Double = 3.7,
    val startupBatteryLevel: Double,
) extends AbstractLocalAction[T](node) {

  private lazy val allocator = node.getProperties
    .stream()
    .filter(_.isInstanceOf[AllocatorProperty[T, P]])
    .findFirst()
    .get()
    .asInstanceOf[AllocatorProperty[T, P]]
  private var previousTime: Double = 0.0

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = {}
}
