package learning.model

case class Component(name: String)
case class Device(name: String)
case class PairComponentDevice(component: Component, device: Device)

case class ActionSpace(componentsCardinality: Int, devicesCardinality: Int){

  private val components = Range(0, componentsCardinality)
    .map(i => Component(s"Component-${i}"))

  private val devices = Range(0, devicesCardinality)
    .map(i => Device(s"Device-${i}"))
    .prepended(Device("MySelf"))

  private val pairs = components.map(c => devices.map(d => PairComponentDevice(c, d)))

  val actions: Seq[List[PairComponentDevice]] =
    pairs.foldLeft(Seq(List.empty[PairComponentDevice])) {
      (acc, seq) => for {
        combo <- acc
        element <- seq
      } yield combo :+ element
    }
}