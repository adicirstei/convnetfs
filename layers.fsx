#load "vol.fsx"

open Convnet

module Input =
  let backward (core:InputCore) = 
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
  let backward (core:SigmoidCore) = 
    let v2 = core.outAct

    { core with
        inAct = Array3D.map (fun (w,dw) -> (w, w * (1.0 - w) * dw )) v2 
    }

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


module Relu =
  let backward (core:ReluCore) = 
    let v2 = core.outAct

    { core with
        inAct = Array3D.map (fun (w,dw) -> (w, (if dw <= 0.0 then 0.0 else dw ))) v2 
    }

  let forward (core:ReluCore) (vol:Vol) (training:bool) = 
    { core with 
        inAct = vol
        outAct = Array3D.map (fun (w, dw) -> (if w < 0.0 then 0.0 else w ), dw ) vol
    }

  let getParamsAndGrads (core:ReluCore) = 
    []

  let create outSx outSy outDepth = 
    ReluLayer {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads

      outDepth = outDepth
      outSx = outSx
      outSy = outSy

      inAct = emptyVol
      outAct = emptyVol
    }

module Maxout =
  let backward (core:MaxoutCore) = 
    if core.outSx = 1 && core.outSx = 1 then
      let a = Array3D.map (fun (w, dw) -> (w, 0.0)) core.inAct
      Array.iteri (fun i sw -> a.[0,0,sw] <- (fst a.[0,0,sw], snd core.outAct.[0,0,i]) ) core.switches

      { core with 
          inAct = a
      }
    else
      core


  let forward (core:MaxoutCore) (vol:Vol) (training:bool) = 
    { core with 
        inAct = vol
        outAct = Array3D.map (fun (w, dw) -> (if w < 0.0 then 0.0 else w ), dw ) vol
    }

  let getParamsAndGrads (core:MaxoutCore) = 
    []

  let create inSx inSy inDepth groupSize = 
    let od = int (inDepth / groupSize)
    MaxoutLayer {

      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads


      groupSize = groupSize
      outSx = inSx
      outSy = inSy

      outDepth = od
      switches = Array.zeroCreate<int> (inSx * inSy * od)

      inAct = emptyVol
      outAct = emptyVol
    }
