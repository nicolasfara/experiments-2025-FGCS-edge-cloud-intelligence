package learning.model

case class Component(id: String)
sealed trait Device
case class MySelf() extends Device
case class EdgeServer(id: Int) extends Device

case class PairComponentDevice(component: Component, device: Device)

case class ActionSpace(components: Seq[Component], devicesCardinality: Int){

  private val devices = Range(0, devicesCardinality)
    .map(i => EdgeServer(i))
    .prepended(MySelf())

  private val pairs = components.map(c => devices.map(d => PairComponentDevice(c, d)))

  val actions: Seq[List[PairComponentDevice]] =
    pairs.foldLeft(Seq(List.empty[PairComponentDevice])) {
      (acc, seq) => for {
        combo <- acc
        element <- seq
      } yield combo :+ element
    }
}