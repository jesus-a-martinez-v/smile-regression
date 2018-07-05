package rbfnet
import smile.data.{Attribute, AttributeDataset, NumericAttribute}
import smile.math.distance.EuclideanDistance
import smile.read
import smile.regression.rbfnet
import smile.util.gaussrbf
import smile.validation.rmse

object RadialBasisFunctionNetwork extends App {
  val attributes = new Array[Attribute](4)

  attributes(0) = new NumericAttribute("AT")
  attributes(1) = new NumericAttribute("V")
  attributes(2) = new NumericAttribute("AP")
  attributes(3) = new NumericAttribute("RH")

  val y = new NumericAttribute("PE")

  val dataFileUri = this.getClass.getClassLoader.getResource("rbfnet/Folds5x2_pp.csv").toURI.getPath
  val data: AttributeDataset = read.csv(dataFileUri, attributes = attributes, response = Some((y, 4)), header = true)

  println("Sneak peek of the data:")
  println(data)

  val centers = new Array[Array[Double]](50)
  val basis = gaussrbf(data.x(), centers)
  val model = rbfnet(
    data.x(),
    data.y(),
    new EuclideanDistance,
    basis,
    centers)
  val predictions = model.predict(data.x())
  println(s"Model's RMSE on the training set: ${rmse(data.y(), predictions)}")

}
