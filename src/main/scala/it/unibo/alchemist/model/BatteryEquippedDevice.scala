package it.unibo.alchemist.model

import it.unibo.alchemist.model.BatteryEquippedDevice.{BATTERY_CAPACITY_MOLECULE, BATTERY_CAPACITY_PERCENTAGE_MOLECULE, getBatteryPercentage}
import it.unibo.alchemist.model.actions.AbstractLocalAction
import it.unibo.alchemist.model.molecules.SimpleMolecule
import org.apache.commons.math3.random.RandomGenerator

/**
 * Action associated to a device with battery consumption.
 *
 * Given a [batteryCapacity] in mAh, [deviceEnergyPerInstruction] in nJ, and [batteryVoltage] in V,
 * this action will consume energy from the battery based on the [softwareComponentsInstructions].
 * This action will also recharge the battery at a rate of [rechargeRate] C-rates.
 *
 * Initially, the battery is charged with [startupCharge] %. By default, it is 100%.
 * By default, the [batteryVoltage] is 3.7 V, and the [rechargeRate] is 0.5 C-rates.
 */
class BatteryEquippedDevice[T, P <: Position[P]](
    private val environment: Environment[T, P],
    private val random: RandomGenerator,
    private val node: Node[T],
    private val batteryCapacity: Double, // Battery capacity in mAh
    private val deviceEnergyPerInstruction: Double, // Energy consumed per instruction in nJ
    private val softwareComponentsInstructions: Map[String, Long], // Instructions per software component
    private val startupCharge: Double = 100.0, // Initial battery charge in %
    private val batteryVoltage: Double = 3.7, // Battery voltage in V
    private val rechargeRate: Double = 0.5 // Recharge rate in C-rates
) extends AbstractLocalAction[T](node) {

    private var previousTimeCache: Option[Double] = None
    private var actualComponents = softwareComponentsInstructions
    private var currentBatteryCapacity = batteryCapacity * startupCharge / 100.0
    private var isRecharging = false

    override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

    override def execute(): Unit = {
        previousTimeCache match {
            case None => previousTimeCache = Some(currentSimulationTime())
            case Some(previousTime) =>
                val deltaTime = currentSimulationTime() - previousTime
                previousTimeCache = Some(currentSimulationTime())
                if (isRecharging) rechargeLogic(deltaTime) else dischargeLogic(deltaTime)
                node.setConcentration(BATTERY_CAPACITY_MOLECULE, currentBatteryCapacity.asInstanceOf[T])
                node.setConcentration(BATTERY_CAPACITY_PERCENTAGE_MOLECULE, getBatteryCapacityPercentage.asInstanceOf[T])
        }
    }

    def getBatteryCapacity: Double = currentBatteryCapacity

    def getBatteryCapacityPercentage: Double = currentBatteryCapacity / batteryCapacity * 100.0

    def recharge: Unit = {
        isRecharging = true
    }

    private def removeComponentExecution(component: String): Unit = {
        softwareComponentsInstructions.get(component) match {
            case Some(_) => actualComponents = actualComponents - component
            case None => throw new IllegalStateException(s"Component $component not found in ${softwareComponentsInstructions.keys}")
        }
    }

    def updateComponentsExecution(allocation: Map[String, Int]): Unit = {
        require(softwareComponentsInstructions.keySet.union(allocation.keySet).size == softwareComponentsInstructions.keySet.size)
        allocation
          .foreach {
              case (component, id) if id == node.getId => addComponentExecution(component)
              case (component, _) => removeComponentExecution(component)
          }
    }

    private def addComponentExecution(component: String): Unit = {
        softwareComponentsInstructions.get(component) match {
            case Some(instructions) => actualComponents = actualComponents + (component -> instructions)
            case None => throw new IllegalStateException(s"Component $component not found in ${softwareComponentsInstructions.keys}")
        }
    }

    private def dischargeLogic(deltaTime: Double): Unit = {
        val epiInJoule = deviceEnergyPerInstruction * 1e-9 // Convert nJ to J
        val componentsConsumedEnergy = actualComponents
          .map { case (component, instructions) =>
            component match {
                case "os" => component -> (random.nextDouble() * instructions * epiInJoule)
                case _ => component -> (instructions * epiInJoule)
            }
          }
          .values
          .reduceOption(_ + _)
          .getOrElse(0.0)
        val componentsConsumedPower = joulesToWatt(componentsConsumedEnergy, deltaTime)
        val batteryConsumedCurrent = wattToMilliAmpere(componentsConsumedPower, batteryVoltage)
        currentBatteryCapacity = (currentBatteryCapacity - batteryConsumedCurrent).max(0.0)
    }

    private def rechargeLogic(deltaTime: Double): Unit = {
        val rechargeCurrent = batteryCapacity * rechargeRate * deltaTime / 3600.0 // mAh
        currentBatteryCapacity = (currentBatteryCapacity + rechargeCurrent).min(batteryCapacity)
        if (currentBatteryCapacity == batteryCapacity) {
            isRecharging = false
        }
    }

    private def joulesToWatt(joules: Double, deltaTime: Double): Double = joules / deltaTime

    private def wattToMilliAmpere(watt: Double, voltage: Double): Double = watt / voltage * 1000

    private def currentSimulationTime(): Double = environment.getSimulation.getTime.toDouble
}

object BatteryEquippedDevice {
    val BATTERY_CAPACITY_MOLECULE = new SimpleMolecule("batteryCapacity")
    val BATTERY_CAPACITY_PERCENTAGE_MOLECULE = new SimpleMolecule("batteryPercentage")

    def getBatteryCapacity[T](node: Node[T]): Double = node.getConcentration(BATTERY_CAPACITY_MOLECULE).asInstanceOf[Double]

    def getBatteryPercentage[T](node: Node[T]): Double = node.getConcentration(BATTERY_CAPACITY_PERCENTAGE_MOLECULE).asInstanceOf[Double]
}
