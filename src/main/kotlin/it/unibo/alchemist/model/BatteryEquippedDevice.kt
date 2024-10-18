package it.unibo.alchemist.model

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
class BatteryEquippedDevice<T, P : Position<P>> @JvmOverloads constructor(
    private val environment: Environment<T, P>,
    private val random: RandomGenerator,
    private val node: Node<T>,
    private val batteryCapacity: Double, // Battery capacity in mAh
    private val deviceEnergyPerInstruction: Double, // Energy consumed per instruction in nJ
    private val softwareComponentsInstructions: Map<String, Long>, // Instructions per software component
    private val startupCharge: Double = 100.0, // Initial battery charge in %
    private val batteryVoltage: Double = 3.7, // Battery voltage in V
    private val rechargeRate: Double = 0.5 // Recharge rate in C-rates
) : AbstractLocalAction<T>(node) {

    @JvmOverloads constructor(
        environment: Environment<T, P>,
        random: RandomGenerator,
        node: Node<T>,
        batteryCapacity: Double, // Battery capacity in mAh
        deviceEnergyPerInstruction: Double, // Energy consumed per instruction in nJ
        programInstructions: Long, // Instructions
        startupCharge: Double = 100.0, // Initial battery charge in %
        batteryVoltage: Double = 3.7, // Battery voltage in V
        rechargeRate: Double = 0.5 // Recharge rate in C-rates
    ) : this(
        environment,
        random,
        node,
        batteryCapacity,
        deviceEnergyPerInstruction,
        mapOf("program" to programInstructions),
        startupCharge,
        batteryVoltage,
        rechargeRate
    )

    private var previousTime = 0.0
    private var actualComponents = softwareComponentsInstructions
    private var currentBatteryCapacity = batteryCapacity * startupCharge / 100.0
    private var isRecharging = false

    override fun cloneAction(node: Node<T>?, reaction: Reaction<T>?): Action<T> = TODO("Not yet implemented")

    @Suppress("UNCHECKED_CAST")
    override fun execute() {
        val deltaTime = currentSimulationTime() - previousTime
        previousTime = currentSimulationTime()
        when {
            isRecharging -> rechargeLogic(deltaTime)
            else -> dischargeLogic(deltaTime)
        }
        node.setConcentration(BATTERY_CAPACITY_MOLECULE, currentBatteryCapacity as T)
        node.setConcentration(BATTERY_CAPACITY_PERCENTAGE_MOLECULE, getBatteryCapacityPercentage() as T)
    }

    fun getBatteryCapacity(): Double = currentBatteryCapacity
    fun getBatteryCapacityPercentage(): Double = currentBatteryCapacity / batteryCapacity * 100.0
    fun recharge() {
        isRecharging = true
    }
    fun removeComponentExecution(component: String) {
        softwareComponentsInstructions[component]?.let {
            actualComponents = actualComponents - component
        } ?: error("Component $component not found in ${softwareComponentsInstructions.keys}")
    }
    fun addComponentExecution(component: String) {
        softwareComponentsInstructions[component]?.let {
            actualComponents = actualComponents + (component to it)
        } ?: error("Component $component not found in ${softwareComponentsInstructions.keys}")
    }

    private fun dischargeLogic(deltaTime: Double) {
        val epiInJoule = deviceEnergyPerInstruction * 1e-9 // Convert nJ to J
        val componentsConsumedEnergy = actualComponents
            .mapValues { (component, instructions) ->
                when (component) {
                    "os" -> random.nextDouble() * instructions * epiInJoule
                    else -> instructions * epiInJoule
                }
            }
            .values
            .reduceOrNull(Double::plus) ?: 0.0
        val componentsConsumedPower = joulesToWatt(componentsConsumedEnergy, deltaTime)
        val batteryConsumedCurrent = wattToMilliAmpere(componentsConsumedPower, batteryVoltage)
        currentBatteryCapacity = (currentBatteryCapacity - batteryConsumedCurrent).coerceAtLeast(0.0)
    }

    private fun rechargeLogic(deltaTime: Double) {
        val rechargeCurrent = batteryCapacity * rechargeRate * deltaTime / 3600.0 // mAh
        currentBatteryCapacity = (currentBatteryCapacity + rechargeCurrent).coerceAtMost(batteryCapacity)
        if (currentBatteryCapacity == batteryCapacity) {
            isRecharging = false
        }
    }

    private fun joulesToWatt(joules: Double, deltaTime: Double): Double = joules / deltaTime
    private fun wattToMilliAmpere(watt: Double, voltage: Double): Double = watt / voltage * 1000
    private fun currentSimulationTime(): Double = environment.simulation.time.toDouble()

    companion object {
        private val BATTERY_CAPACITY_MOLECULE = SimpleMolecule("CurrentBatteryCapacity")
        private val BATTERY_CAPACITY_PERCENTAGE_MOLECULE = SimpleMolecule("CurrentBatteryPercentage")
        @JvmStatic
        fun <T> getBatteryCapacity(node: Node<T>): Double = node.getConcentration(BATTERY_CAPACITY_MOLECULE) as Double
        @JvmStatic
        fun <T> getBatteryCapacityPercentage(node: Node<T>): Double = node.getConcentration(BATTERY_CAPACITY_PERCENTAGE_MOLECULE) as Double
//        fun <T> Node<T>.getBatteryCapacity(): Double = getConcentration(BATTERY_CAPACITY_MOLECULE) as Double
//        fun <T> Node<T>.getBatteryCapacityPercentage(): Double = getConcentration(BATTERY_CAPACITY_PERCENTAGE_MOLECULE) as Double
    }
}
