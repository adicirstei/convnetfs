#load "vol.fsx"

open Convnet

module Input =
  let backward core = 
    core

  let forward (core:InputCore) (vol:Vol) (training:bool) = 
    { core with 
        inAct = vol
        outAct = vol
    }

  let getParamsAndGrads (core:InputCore) = 
    []

  let create outDepth outSx outSy = 
    InputLayer {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads
      outDepth = outDepth
      outSx = outSx
      outSy = outSy
      inAct = emptyVol
      outAct = emptyVol
    }

module Sigmoid =
  let backward core = 
    core

  let forward (core:SigmoidCore) (vol:Vol) (training:bool) = 
    let v2 = Array3D.map (fun (w, dw) -> 1.0 / (1.0 + exp (-w)), dw ) vol
    { core with 
        inAct = vol
        outAct = v2
    }

  let getParamsAndGrads (core:SigmoidCore) = 
    []

  let create outSx outSy outDepth = 
    SigmoidLayer {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads

      outDepth = outDepth
      outSx = outSx
      outSy = outSy

      inAct = emptyVol
      outAct = emptyVol
    }
