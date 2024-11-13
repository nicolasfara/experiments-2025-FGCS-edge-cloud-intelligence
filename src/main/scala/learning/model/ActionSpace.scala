package learning.model

/**
 * For each component we have a set of possible actions C0_actions = {C0_A, C0_ES1, ..., C0_ESn}
 * The complete action space is the cartesian product {(C0_A,C1_A), (C0_A, C1_ES1), ...}
 * The cardinality of the action space is |C0_actions| * ... * |Ck_actions|
 * */

sealed trait Device
case class Component(id: String)
case class MySelf() extends Device
case class EdgeServer(id: Int) extends Device
case class Cloud(id: Int) extends Device

case class PairComponentDevice(component: Component, device: Device)

case class ActionSpace(components: Seq[Component], edgeServerCardinality: Int, cloudCardinality: Int){

  private val edgeServer: Seq[Device] = Range(0, edgeServerCardinality)
    .map(EdgeServer)

  private val cloud: Seq[Device] = Range(0, cloudCardinality)
    .map(Cloud)

  private val devices: Seq[Device] = (edgeServer ++ cloud)
    .appended(MySelf())

  private val pairs = components.map(c => devices.map(d => PairComponentDevice(c, d)))

  val actions: Seq[List[PairComponentDevice]] =
    pairs.foldLeft(Seq(List.empty[PairComponentDevice])) {
      (acc, seq) => for {
        combo <- acc
        element <- seq
      } yield combo :+ element
    }
}
object Main extends App {
  val as =   ActionSpace(Seq(Component("Gradient"), Component("Greater")), 3, 2)
  as.actions.zipWithIndex.foreach(println(_))
  println(as.actions.size)
}