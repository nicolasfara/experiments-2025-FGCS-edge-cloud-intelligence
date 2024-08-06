package it.unibo.alchemist.model

sealed trait Target
final case class SurrogateNode(kind: String) extends Target
case object LocalNode extends Target

class AllocatorProperty[T, P <: Position[P]](
    environment: Environment[T, _],
    node: Node[T],
    startAllocation: Map[String, Target]
) extends NodeProperty[T] {

  def getComponentsAllocation: Map[String, Target] = ???

  def getPhysicalComponentsAllocations: Map[String, Node[T]] = ???

  def setComponentAllocation(component: String, target: Target): Unit = ???

  def checkAllocation(): Unit = ???

  override def getNode: Node[T] = node

  override def cloneOnNewNode(node: Node[T]): NodeProperty[T] = ???
}
