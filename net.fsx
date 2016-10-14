#load "layers.fsx"

open Convnet


type Net = {
  layers : LayerType list
}

let backward net y =
  let ll :: rest = List.rev net.layers
  let core, (Some loss) = ll.backward ll.data (Some y)

  let newLL = 
    { ll with data = core} ::
    List.map (fun l -> 
      let c, lo =  l.backward l.data None
      { l with data = c}
      )  rest
    |> List.rev
  { net with layers = newLL }, loss

let forward net x = 
  let state : LayerType list * Vol = [], x
  let newLL, act = 
    List.fold (fun (ll, prevAct) l -> 
      let c, a = l.forward l.data prevAct
      (ll @ [ { l with data = c } ]), a) state net.layers
  { net with layers = newLL }, act

let getParamsAndGrads net = 
  net.layers
  |> List.collect (fun l -> 
      l.getParamsAndGrads l.data
  )