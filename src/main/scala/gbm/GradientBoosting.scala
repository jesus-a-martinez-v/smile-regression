package gbm

import smile.data.{Attribute, AttributeDataset, NumericAttribute}
import smile.read
import smile.regression.gbm
import smile.validation.rmse

object GradientBoosting extends App {
  val attributes = new Array[Attribute](5)

  attributes(0) = new NumericAttribute("Frequency")
  attributes(1) = new NumericAttribute("Angle of attack")
  attributes(2) = new NumericAttribute("Chord length")
  attributes(3) = new NumericAttribute("Free-stream velocity")
  attributes(4) = new NumericAttribute("Suction side displacement thickness")

  val y = new NumericAttribute("Scaled sound pressure level")

  val dataFileUri = this.getClass.getClassLoader.getResource("gbm/airfoil_self_noise.dat").toURI.getPath
  val data: AttributeDataset = read.table(dataFileUri, attributes = attributes, response = Some((y, 5)))

  println("Sneak peek of the data:")
  println(data)

  val model = gbm(data.x(), data.y())
  val predictions = model.predict(data.x())

  println(s"Model's RMSE on the training set: ${rmse(data.y(), predictions)}")

}
