#load "layers.fsx"

open Convnet


type Net = {
  layers : LayerType list
}

let forward net x t = net

let getParamsAndGrads net = 
  net.layers
  |> List.collect (fun l -> 
      l.getParamsAndGrads l.data
  )