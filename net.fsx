#load "layers.fsx"

type Net = {
  layers : Convnet.LayerType list
}

let forward net x t = net