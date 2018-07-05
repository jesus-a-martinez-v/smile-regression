package dt

import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.read
import smile.regression.cart
import smile.validation.rmse

object RegressionTree extends App {
  val attributes = new Array[Attribute](8)

  attributes(0) = new NominalAttribute("cylinders")
  attributes(1) = new NumericAttribute("displacement")
  attributes(2) = new NumericAttribute("horsepower")
  attributes(3) = new NumericAttribute("weight")
  attributes(4) = new NumericAttribute("acceleration")
  attributes(5) = new NominalAttribute("model year")
  attributes(6) = new NominalAttribute("origin")
  attributes(7) = new NominalAttribute("car name")

  val y = new NumericAttribute("mpg")

  val dataFileUri = this.getClass.getClassLoader.getResource("dt/auto-mpg.data-original").toURI.getPath
  val data: AttributeDataset = read.table(dataFileUri, attributes = attributes, response = Some((y, 0)))

  println("Sneak peek of the data:")
  println(data)

  val model = cart(data.x(), data.y(), maxNodes = 25)
  val predictions = model.predict(data.x())

  println(s"Model's RMSE on the training set: ${rmse(data.y(), predictions)}")
}
