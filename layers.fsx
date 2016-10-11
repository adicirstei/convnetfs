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
    let a = Array3D.map (fun (w, dw) -> (w, 0.0)) core.inAct
    Array3D.iteri (fun x y d (w,dw) -> a.[x,y,d] <- w, 0.0) a
    if core.outSx = 1 && core.outSy = 1 then
      
      Array.iteri (fun i sw -> a.[0,0,sw] <- (fst a.[0,0,sw], snd core.outAct.[0,0,i]) ) core.switches

      { core with 
          inAct = a
      }
    else
      Array3D.iteri (fun x y d (w, dw) -> 
                      let sw = core.switches.[(core.outSx * core.outSy *d) + x * core.outSy + y]
                      a.[x,y,sw] <- (fst a.[x,y,sw]) , (snd core.outAct.[x,y,d])  ) core.outAct
      { core with 
          inAct = a
      }


  let forward (core:MaxoutCore) (v:Vol) (training:bool) = 
    let v2 = Vol.constCreate core.outSx core.outSy core.outDepth 0.0
    
    Array3D.iteri (fun x y i (w,dw) -> 
      let ix = i * core.groupSize
      let ((a,wa),ai) = Vol.maxDim3 v x y [ix..ix+core.groupSize-1]

      Vol.set v2 x y i a
      core.switches.[core.outSx * core.outSy * i + y * core.outSx + y] <- ai
    ) core.inAct
 

    { core with 
        inAct = v
        outAct = Array3D.map (fun (w, dw) -> (if w < 0.0 then 0.0 else w ), dw ) v
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

module Tanh =
  let backward (core:TanhCore) = 
    let v2 = core.outAct

    { core with
        inAct = Array3D.map (fun (w,dw) -> (w, (1.0 - w * w) * dw )) v2 
    }

  let forward (core:TanhCore) (vol:Vol) (training:bool) = 
    let v2 = Array3D.map (fun (w, dw) -> tanh w, dw ) (Vol.cloneAndZero vol)
    { core with 
        inAct = vol
        outAct = v2
    }

  let getParamsAndGrads (core:TanhCore) = 
    []

  let create outSx outSy outDepth = 
    TanhLayer {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads

      outDepth = outDepth
      outSx = outSx
      outSy = outSy

      inAct = emptyVol
      outAct = emptyVol
    }

module FullyConnected =
  let backward (core:FullyConnCore) = 
    let v2 = core.outAct

    { core with
        inAct = Array3D.map (fun (w,dw) -> (w, (if dw <= 0.0 then 0.0 else dw ))) v2 
    }

  let forward (core:FullyConnCore) (vol:Vol) (training:bool) = 
    { core with 
        inAct = vol
        outAct = Array3D.map (fun (w, dw) -> (if w < 0.0 then 0.0 else w ), dw ) vol
    }

  let getParamsAndGrads (core:FullyConnCore) = 
    []
    
  let create outDepth l1DecMul l2DecMul inSx inSy inDepth bias = 
    FullyConnLayer {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads

      outDepth = outDepth

      numInputs = inSx * inSy * inDepth
      outSx = 1
      outSy = 1
      l1DecayMul = l1DecMul
      l2DecayMul = l2DecMul

      filters = [1..outDepth] |> List.map (fun _ ->  Vol.randCreate 1 1 (inSx * inSy * inDepth))
      biases = Vol.constCreate 1 1 outDepth bias

      inAct = emptyVol
      outAct = emptyVol
    }
