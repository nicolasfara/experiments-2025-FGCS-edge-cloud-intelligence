package it.unibo.alchemist.model.implementations.actions

import it.unibo.alchemist.model.{Action, Dependency, Environment, Molecule, Node, Position, Reaction}
import it.unibo.alchemist.model.actions.AbstractLocalAction
import it.unibo.alchemist.model.implementations.actions.RunScafiProgram.NeighborData
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist
import it.unibo.alchemist.model.{Time => AlchemistTime, _}
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.{CONTEXT, EXPORT, ID, Path, _}
import it.unibo.alchemist.scala.PimpMyAlchemist._
import it.unibo.scafi.space.Point3D
import org.apache.commons.math3.random.RandomGenerator
import org.apache.commons.math3.util.FastMath
import org.kaikikm.threadresloader.ResourceLoader

import java.util.concurrent.TimeUnit
import scala.collection.mutable
import scala.concurrent.duration.FiniteDuration
import scala.jdk.CollectionConverters.{CollectionHasAsScala, IteratorHasAsScala}
import scala.util.{Failure, Try}

sealed abstract class RunScafiProgram[T, P <: Position[P]](node: Node[T]) extends AbstractLocalAction[T](node) {
  def asMolecule: Molecule = new SimpleMolecule(getClass.getSimpleName)
  def isComputationalCycleComplete: Boolean
  def programNameMolecule: Molecule
  def programDag: Map[String, List[String]]
  def prepareForComputationalCycle(): Unit
}
object RunScafiProgram {
  case class NeighborData[P <: Position[P]](exportData: EXPORT, position: P, executionTime: AlchemistTime)

  implicit class RichMap[K, V](map: Map[K, V]) {
    def mapValuesStrict[T](f: V => T): Map[K, T] = map.map(tp => tp._1 -> f(tp._2))
  }
}

final class RunApplicationScafiProgram[T, P <: Position[P]](
    environment: Environment[T, P],
    node: Node[T],
    reaction: Reaction[T],
    randomGenerator: RandomGenerator,
    programName: String,
    retentionTime: Double,
    programDagMapping: Map[String, List[String]] = Map.empty
) extends RunScafiProgram[T, P](node) {

  def this(
      environment: Environment[T, P],
      node: Node[T],
      reaction: Reaction[T],
      randomGenerator: RandomGenerator,
      programName: String
  ) = this(environment, node, reaction, randomGenerator, programName, FastMath.nextUp(1.0 / reaction.getTimeDistribution.getRate))

  declareDependencyTo(Dependency.EVERY_MOLECULE)

  import RunScafiProgram.NeighborData
  val program = ResourceLoader
    .classForName(programName)
    .getDeclaredConstructor()
    .newInstance()
    .asInstanceOf[CONTEXT => EXPORT]
  override val programDag: Map[String, List[String]] = programDagMapping
  override val programNameMolecule = new SimpleMolecule(programName)
  lazy val nodeManager = new SimpleNodeManager(node)
  private var neighborhoodManager: Map[ID, NeighborData[P]] = Map()
  private val commonNames = new ScafiIncarnationForAlchemist.StandardSensorNames {}
  private val targetMolecule = new SimpleMolecule("Target")
  private var completed = false
  private lazy val allocatorProperty: Option[AllocatorProperty[T, P]] = node.getProperties.asScala
    .find(_.isInstanceOf[AllocatorProperty[T, P]])
    .map(_.asInstanceOf[AllocatorProperty[T, P]])

  private val inputFromComponents = collection.mutable.Map[ID, mutable.Buffer[(Path, T)]]()

  override def cloneAction(node: Node[T], reaction: Reaction[T]) =
    new RunApplicationScafiProgram(environment, node, reaction, randomGenerator, programName, retentionTime)

  implicit def euclideanToPoint(point: P): Point3D = point.getDimensions match {
    case 1 => Point3D(point.getCoordinate(0), 0, 0)
    case 2 => Point3D(point.getCoordinate(0), point.getCoordinate(1), 0)
    case 3 => Point3D(point.getCoordinate(0), point.getCoordinate(1), point.getCoordinate(2))
  }

  private def isOffloadedToSurrogate: Boolean = {
    val result = for {
      allocator <- allocatorProperty
      targetHostKind <- allocator.getPhysicalComponentsAllocations.get(asMolecule.getName)
    } yield targetHostKind != node
    result.getOrElse(false)
  }

  override def execute(): Unit = {
    import scala.jdk.CollectionConverters._
    val position: P = environment.getPosition(node)
    // NB: We assume it.unibo.alchemist.model.Time = DoubleTime
    //     and that its "time unit" is seconds, and then we get NANOSECONDS
    val alchemistCurrentTime = Try(environment.getSimulation)
      .map(_.getTime)
      .orElse(Failure(new IllegalStateException("The simulation is uninitialized (did you serialize the environment?)")))
      .get
    val currentTime: Long = alchemistTimeToNanos(alchemistCurrentTime)
    manageRetentionMessages(alchemistCurrentTime)

    // ----- Create context
    // Add self node to the neighborhood manager
    neighborhoodManager = neighborhoodManager.updatedWith(node.getId) {
      case value @ Some(_) => value
      case None            => Some(NeighborData(factory.emptyExport(), position, Double.NaN))
    }
    val deltaTime: Long =
      currentTime - neighborhoodManager.get(node.getId).map(d => alchemistTimeToNanos(d.executionTime)).getOrElse(0L)
    val localSensors = node.getContents.asScala.map { case (k, v) => k.getName -> v }
    val neighborhoodSensors = scala.collection.mutable.Map[CNAME, Map[ID, Any]]()
    val exports: Iterable[(ID, EXPORT)] = neighborhoodManager.view.mapValues(_.exportData)
    val context = buildContext(exports, localSensors.toMap, neighborhoodSensors, alchemistCurrentTime, deltaTime, currentTime, position)

    mergeInputFromComponentsWithExport()

    // ----- Check if the program is offloaded to a surrogate or not
    if (isOffloadedToSurrogate) {
      // Check if the program is offloaded to a surrogate
      for {
        allocator <- allocatorProperty
        targetHostKind <- allocator.getComponentsAllocation.get(asMolecule.getName)
        if targetHostKind != LocalNode
        surrogateNode <- allocator.getPhysicalComponentsAllocations.get(asMolecule.getName) // Where is physical executed this program? (Node ID)
        surrogateProgram <- SurrogateScafiIncarnation
          .allScafiProgramsForType(surrogateNode, classOf[RunSurrogateScafiProgram[T, P]])
          .map(_.asInstanceOf[RunSurrogateScafiProgram[T, P]])
          .find(_.asMolecule == asMolecule)
      } {
        //        println(s"Node ${node.getId} has forward to $targetHostKind with id ${surrogateNode.getId}")
        surrogateProgram.setContextFor(node.getId, context)
        surrogateProgram.setCurrentNeighborhoodOf(node.getId, currentApplicativeNeighborhood)
      }
    } else {
      // Execute normal program since is executed locally
      val computed = program(context)
      val toSend = NeighborData(computed, position, alchemistCurrentTime)
      neighborhoodManager = neighborhoodManager + (node.getId -> toSend)
    }
    for {
      programResult <- neighborhoodManager.get(node.getId)
      result <- programResult.exportData.get[T](factory.emptyPath())
    } node.setConcentration(programNameMolecule, result)
    completed = true
  }

  private def currentApplicativeNeighborhood: Set[ID] = {
    environment
      .getNeighborhood(node)
      .getNeighbors
      .iterator()
      .asScala
      .filter(_.getConcentration(targetMolecule) == LocalNode.asInstanceOf[T])
      .map(_.getId)
      .toSet
  }

  private def alchemistTimeToNanos(time: AlchemistTime): Long = (time.toDouble * 1_000_000_000).toLong

  private def buildContext(
      exports: Iterable[(ID, EXPORT)],
      localSensors: Map[String, T],
      neighborhoodSensors: scala.collection.mutable.Map[CNAME, Map[ID, Any]],
      alchemistCurrentTime: AlchemistTime,
      deltaTime: Long,
      currentTime: Long,
      position: P
  ): CONTEXT = new ContextImpl(node.getId, exports, localSensors, Map.empty) {
    override def nbrSense[TT](nsns: CNAME)(nbr: ID): Option[TT] =
      neighborhoodSensors
        .getOrElseUpdate(
          nsns,
          nsns match {
            case commonNames.NBR_LAG =>
              neighborhoodManager.mapValuesStrict[FiniteDuration](nbr =>
                FiniteDuration(alchemistTimeToNanos(alchemistCurrentTime - nbr.executionTime), TimeUnit.NANOSECONDS)
              )
            /*
             * nbrDelay is estimated: it should be nbr(deltaTime), here we suppose the round frequency
             * is negligibly different between devices.
             */
            case commonNames.NBR_DELAY =>
              neighborhoodManager.mapValuesStrict[FiniteDuration](nbr =>
                FiniteDuration(
                  alchemistTimeToNanos(nbr.executionTime) + deltaTime - currentTime,
                  TimeUnit.NANOSECONDS
                )
              )
            case commonNames.NBR_RANGE => neighborhoodManager.mapValuesStrict[Double](_.position.distanceTo(position))
            case commonNames.NBR_VECTOR =>
              neighborhoodManager.mapValuesStrict[Point3D](_.position.minus(position.getCoordinates))
            case NBR_ALCHEMIST_LAG =>
              neighborhoodManager.mapValuesStrict[Double](alchemistCurrentTime - _.executionTime)
            case NBR_ALCHEMIST_DELAY =>
              neighborhoodManager.mapValuesStrict(nbr => alchemistTimeToNanos(nbr.executionTime) + deltaTime - currentTime)
          }
        )
        .get(nbr)
        .map(_.asInstanceOf[TT])

    override def sense[TT](lsns: String): Option[TT] = (lsns match {
      case LSNS_ALCHEMIST_COORDINATES  => Some(position.getCoordinates)
      case commonNames.LSNS_DELTA_TIME => Some(FiniteDuration(deltaTime, TimeUnit.NANOSECONDS))
      case commonNames.LSNS_POSITION =>
        val k = position.getDimensions
        Some(
          Point3D(
            position.getCoordinate(0),
            if (k >= 2) position.getCoordinate(1) else 0,
            if (k >= 3) position.getCoordinate(2) else 0
          )
        )
      case commonNames.LSNS_TIMESTAMP  => Some(currentTime)
      case commonNames.LSNS_TIME       => Some(java.time.Instant.ofEpochMilli((alchemistCurrentTime * 1000).toLong))
      case LSNS_ALCHEMIST_NODE_MANAGER => Some(nodeManager)
      case LSNS_ALCHEMIST_DELTA_TIME =>
        Some(
          alchemistCurrentTime.minus(
            neighborhoodManager.get(node.getId).map(_.executionTime).getOrElse(AlchemistTime.INFINITY)
          )
        )
      case LSNS_ALCHEMIST_ENVIRONMENT => Some(environment)
      case LSNS_ALCHEMIST_RANDOM      => Some(randomGenerator)
      case LSNS_ALCHEMIST_TIMESTAMP   => Some(alchemistCurrentTime)
      case _                          => localSensors.get(lsns)
    }).map(_.asInstanceOf[TT])
  }

  def sendExport(id: ID, exportData: NeighborData[P]): Unit = {
    neighborhoodManager += id -> exportData
  }

  def getExport(id: ID): Option[NeighborData[P]] = neighborhoodManager.get(id)

  def isComputationalCycleComplete: Boolean = completed

  override def prepareForComputationalCycle(): Unit = completed = false

  def setResultWhenOffloaded(result: T): Unit = node.setConcentration(asMolecule, result)

  def feedInputFromNode(node: ID, value: (Path, T)): Unit = {
    inputFromComponents.get(node) match {
      case Some(inputs) =>
        val newInputs = inputs.filter(!_._1.matches(value._1))
        inputFromComponents += node -> (newInputs += value)
      case None => inputFromComponents += node -> mutable.Buffer(value)
    }
  }

  def generateComponentOutputField(): (Path, Option[T]) = {
    val path = factory.path(Scope(programName))
    val result = neighborhoodManager(node.getId).exportData.get[T](factory.emptyPath())
    path -> result
  }

  private def mergeInputFromComponentsWithExport(): Unit = {
    for {
      (nodeId, inputs) <- inputFromComponents
      export <- neighborhoodManager.get(nodeId).map(_.exportData)
      (path, value) <- inputs
    } export.put(path, value)
  }

  private def manageRetentionMessages(currentTime: AlchemistTime): Unit = {
    neighborhoodManager = neighborhoodManager.filter { case (id, data) =>
      id == node.getId || data.executionTime >= currentTime - retentionTime
    }
  }
}

final class RunSurrogateScafiProgram[T, P <: Position[P]](
    environment: Environment[T, P],
    node: Node[T],
    reaction: Reaction[T],
    randomGenerator: RandomGenerator,
    programName: String,
    retentionTime: Double,
    programDagMapping: Map[String, List[String]] = Map.empty
) extends RunScafiProgram[T, P](node) {

  def this(
      environment: Environment[T, P],
      node: Node[T],
      reaction: Reaction[T],
      randomGenerator: RandomGenerator,
      programName: String
  ) = this(environment, node, reaction, randomGenerator, programName, FastMath.nextUp(1.0 / reaction.getTimeDistribution.getRate))

  private var completed = false
  declareDependencyTo(Dependency.EVERY_MOLECULE)

  val program = ResourceLoader
    .classForName(programName)
    .getDeclaredConstructor()
    .newInstance()
    .asInstanceOf[CONTEXT => EXPORT]
  override val programDag = programDagMapping
  override val programNameMolecule = new SimpleMolecule(programName)
  private val surrogateForNodes = collection.mutable.Set[ID]()
  private val contextManager = collection.mutable.Map[ID, CONTEXT]()
  // Map of node ID to a map of neighbor ID to NeighborData
  private val neighborhoodManager = collection.mutable.Map[ID, collection.mutable.Map[ID, NeighborData[P]]]()
  private val currentNeighborhoodOfNodes = collection.mutable.Map[ID, Set[ID]]()
  private val targetMolecule = new SimpleMolecule("Target")

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] =
    new RunSurrogateScafiProgram(environment, node, reaction, randomGenerator, programName, retentionTime)

  override def execute(): Unit = {
    val alchemistCurrentTime = Try(environment.getSimulation)
      .map(_.getTime)
      .orElse(Failure(new IllegalStateException("The simulation is uninitialized (did you serialize the environment?)")))
      .get

    // Clean the neighborhood manager according to the retention time
    neighborhoodManager.foreach { case (_, data) =>
      data.filterInPlace((_, p) => p.executionTime >= alchemistCurrentTime - retentionTime)
    }
    // Clean the surrogateForNodes according to the retention time
    val applicativeDevice = activeApplicationNeighborDevices.map(_.getId)
    surrogateForNodes.filterInPlace(nodeId => applicativeDevice.contains(nodeId))
    currentNeighborhoodOfNodes.filterInPlace((nodeId, _) => applicativeDevice.contains(nodeId))

    // Run the program for each node offloading the computation to this surrogate
    surrogateForNodes.foreach(deviceId => {
      contextManager.get(deviceId) match {
        case Some(contextNode) =>
          val computedResult = program(contextNode)
          val nodePosition = environment.getPosition(environment.getNodeByID(deviceId))
          val toSend = NeighborData(computedResult, nodePosition, alchemistCurrentTime)
          val neighborsToSend = currentNeighborhoodOfNodes(deviceId)
            .map(neighId => neighId -> toSend)
            .to(collection.mutable.Map) ++ Map(deviceId -> toSend)
          neighborhoodManager.put(deviceId, neighborsToSend)
          environment.getNodeByID(deviceId).setConcentration(asMolecule, computedResult.root[T]())
        case None => ()
      }
    })
    val resultsForEachComputedNode = neighborhoodManager.view.mapValues(_.view.mapValues(_.exportData.root[T]()).toMap).toMap
    node.setConcentration(asMolecule, resultsForEachComputedNode.asInstanceOf[T])
    node.setConcentration(new SimpleMolecule(s"SurrogateFor[$programName]"), isSurrogateFor.asInstanceOf[T])
    completed = true
  }

  private def activeApplicationNeighborDevices: List[Node[T]] = {
    environment
      .getNeighborhood(node)
      .getNeighbors
      .iterator()
      .asScala
      .filter(_.getConcentration(targetMolecule) == LocalNode.asInstanceOf[T])
      .toList
  }

  def setSurrogateFor(nodeId: ID): Unit = surrogateForNodes.add(nodeId)

  def setCurrentNeighborhoodOf(nodeId: ID, neighborhood: Set[ID]): Unit =
    currentNeighborhoodOfNodes.put(nodeId, neighborhood)

  def removeSurrogateFor(nodeId: ID): Unit = {
    surrogateForNodes.remove(nodeId)
    contextManager.remove(nodeId)
  }

  def isSurrogateForNode(nodeId: ID): Boolean = surrogateForNodes.contains(nodeId)

  def isSurrogateFor: Set[ID] = surrogateForNodes.toSet

  def setContextFor(nodeId: ID, context: CONTEXT): Unit = contextManager.put(nodeId, context)

  def getComputedResultFor(nodeId: ID): Option[NeighborData[P]] =
    for {
      neighbors <- neighborhoodManager.get(nodeId)
      computedResult <- neighbors.get(nodeId)
    } yield computedResult

  def isComputationalCycleComplete: Boolean = completed

  override def prepareForComputationalCycle(): Unit = completed = false
}
