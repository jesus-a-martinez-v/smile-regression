package ridge

import smile.data.{Attribute, AttributeDataset, NominalAttribute, NumericAttribute}
import smile.read
import smile.regression.ridge
import smile.validation.rmse

object RidgeRegression extends App {
  val attributes = new Array[Attribute](9)

  attributes(0) = new NominalAttribute("vendor name")
  attributes(1) = new NominalAttribute("Model Name")
  attributes(2) = new NumericAttribute("MYCT")
  attributes(3) = new NumericAttribute("MMIN")
  attributes(4) = new NumericAttribute("MMAX")
  attributes(5) = new NumericAttribute("CACH")
  attributes(6) = new NumericAttribute("CHMIN")
  attributes(7) = new NumericAttribute("CHMAX")
  attributes(8) = new NumericAttribute("PRP")

  val y = new NumericAttribute("ERP")

  val dataFileUri = this.getClass.getClassLoader.getResource("ridge/machine.data").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((y, 9)))

  println("Sneak peek of the data:")
  println(data)

  val model = ridge(data.x(), data.y(), 0.01)
  val predictions = model.predict(data.x())

  println(s"Model's RMSE on the training set: ${rmse(data.y(), predictions)}")

}
