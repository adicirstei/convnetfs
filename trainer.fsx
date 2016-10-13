#load "net.fsx"

open Convnet


type Method = 
  | Sgd
  | Adam
  | Adagrad
  | Adadelta
  | Windowgrad
  | Netsterov


type Options = {
  learningRate : float
  l1Decay : float
  l2Decay : float
  method : Method
  momentum : float
  ro : float
  eps : float
  beta1 : float
  beta2 : float
  batchSize : int
}
open Net

type State = {
  options : Options
  method : Method
  batchSize : int
  momentum : float
  net : Net
  k : int
  gsum : float [] list
  xsum : float [] list
}

let createTrainer opts net =
  {
    options = opts
    method = opts.method
    batchSize = opts.batchSize
    momentum = opts.momentum
    net = net
    k = 0
    gsum = []
    xsum = []
  }

let batchUpdate state = 
  let pglist = Net.getParamsAndGrads state.net
  let gsum, xsum = 
    pglist
    |> List.map (fun (pg:ParamsGrads) -> 
        if List.isEmpty state.gsum && (state.method <> Sgd || state.momentum > 0.0) then
          Util.zeros pg.params.Length, 
            if state.method = Adam || state.method = Adadelta then 
              Util.zeros pg.params.Length
            else
              [||]
        else 
          [||], [||]
    
    ) 
    |> List.unzip

  { state with
      gsum = gsum
      xsum = xsum
  }


let train state x y = 
  { state with
      net = Net.forward (state.net)  x true 
      k = state.k + 1
  }
  |> (fun s -> if s.k % s.options.batchSize = 0 then batchUpdate s else s)