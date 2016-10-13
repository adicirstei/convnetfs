#load "vol.fsx"

open Convnet

module Input =
  let backward (core:Core) = 
    core
  let forward (core:Core) (vol:Vol) (training:bool) = 
    let (InputCore c) = core
    InputCore { c with 
        inAct = vol
        outAct = vol
    }
  let getParamsAndGrads (core:Core) = 
    []
  let create outDepth outSx outSy = 
    {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads
      data = 
        InputCore {
          outDepth = outDepth
          outSx = outSx
          outSy = outSy
          inAct = emptyVol
          outAct = emptyVol
        }
    }
module Sigmoid =
  let backward c = 
    let (SigmoidCore core) = c
    let v2 = core.outAct

    SigmoidCore { core with
      inAct = Array3D.map (fun (w,dw) -> (w, w * (1.0 - w) * dw )) v2 
    }

  let forward c (vol:Vol) (training:bool) = 
    let (SigmoidCore core) = c
    let v2 = Array3D.map (fun (w, dw) -> 1.0 / (1.0 + exp (-w)), dw ) vol
    SigmoidCore { core with 
        inAct = vol
        outAct = v2
    }

  let getParamsAndGrads (core:Core) = 
    []
  let create outSx outSy outDepth = 
    {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads

      data = 
        SigmoidCore { 
          outDepth = outDepth
          outSx = outSx
          outSy = outSy

          inAct = emptyVol
          outAct = emptyVol
        }
    }


module Relu =
  let backward c = 
    let (ReluCore core) = c
    let v2 = core.outAct

    ReluCore { core with
      inAct = Array3D.map (fun (w,dw) -> (w, (if dw <= 0.0 then 0.0 else dw ))) v2 
    }

  let forward c (vol:Vol) (training:bool) = 
    let (ReluCore core) = c
    ReluCore { core with 
      inAct = vol
      outAct = Array3D.map (fun (w, dw) -> (if w < 0.0 then 0.0 else w ), dw ) vol
    }

  let getParamsAndGrads (core:Core) = 
    []

  let create outSx outSy outDepth = 
    {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads
      data = ReluCore {
        outDepth = outDepth
        outSx = outSx
        outSy = outSy

        inAct = emptyVol
        outAct = emptyVol
      }
    }

module Maxout =
  let backward c = 
    let (MaxoutCore core) = c
    let a = Array3D.map (fun (w, dw) -> (w, 0.0)) core.inAct
    Array3D.iteri (fun x y d (w,dw) -> a.[x,y,d] <- w, 0.0) a
    if core.outSx = 1 && core.outSy = 1 then
      
      Array.iteri (fun i sw -> a.[0,0,sw] <- (fst a.[0,0,sw], snd core.outAct.[0,0,i]) ) core.switches

      MaxoutCore { core with 
        inAct = a
      }
    else
      Array3D.iteri (fun x y d (w, dw) -> 
                      let sw = core.switches.[(core.outSx * core.outSy *d) + x * core.outSy + y]
                      a.[x,y,sw] <- (fst a.[x,y,sw]) , (snd core.outAct.[x,y,d])  ) core.outAct
      MaxoutCore { core with 
        inAct = a
      }


  let forward c (v:Vol) (training:bool) = 
    let (MaxoutCore core) = c
    let v2 = Vol.constCreate core.outSx core.outSy core.outDepth 0.0
    
    Array3D.iteri (fun x y i (w,dw) -> 
      let ix = i * core.groupSize
      let ((a,wa),ai) = Vol.maxDim3 v x y [ix..ix+core.groupSize-1]

      Vol.set v2 x y i a
      core.switches.[core.outSx * core.outSy * i + y * core.outSx + y] <- ai
    ) core.inAct
 

    MaxoutCore { core with 
      inAct = v
      outAct = Array3D.map (fun (w, dw) -> (if w < 0.0 then 0.0 else w ), dw ) v2
    }

  let getParamsAndGrads (core:Core) = 
    []

  let create inSx inSy inDepth groupSize = 
    let od = int (inDepth / groupSize)
    {

      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads

      data = MaxoutCore {
        groupSize = groupSize
        outSx = inSx
        outSy = inSy

        outDepth = od
        switches = Array.zeroCreate<int> (inSx * inSy * od)

        inAct = emptyVol
        outAct = emptyVol
      }
    }

module Tanh =
  let backward c = 
    let (TanhCore core) = c
    let v2 = core.outAct

    TanhCore { core with
      inAct = Array3D.map (fun (w,dw) -> (w, (1.0 - w * w) * dw )) v2 
    }

  let forward c (vol:Vol) (training:bool) = 
    let (TanhCore core) = c
    let v2 = Array3D.map (fun (w, dw) -> tanh w, dw ) (Vol.cloneAndZero vol)
    TanhCore { core with 
      inAct = vol
      outAct = v2
    }

  let getParamsAndGrads (core:Core) = 
    []

  let create outSx outSy outDepth = 
    {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads
      data = TanhCore {
        outDepth = outDepth
        outSx = outSx
        outSy = outSy

        inAct = emptyVol
        outAct = emptyVol
      }
    }

module FullyConnected =
  let backward c = 
    let (FullyConnCore core) = c
    let v = 
      core.inAct
      |> Array3D.map (fun (w,_) -> (w, 0.0))

    FullyConnCore { core with
        inAct = Array3D.mapi (fun x y i (w,dw) -> 
          let chg = snd core.outAct.[x,y,i]
          let tfi = core.filters.[i]
          let tfiw = 
            Vol.getWs tfi 
            |> Vol.flatten
            |> Seq.sumBy (fun w -> w*chg)

          w, dw
        )  v  /// to implement
        filters  = List.mapi (fun i f -> 
          Array3D.mapi (fun x y d (w, dw) -> w, dw + (fst core.inAct.[x,y,d]) * (snd core.outAct.[0,0,i])) f
        ) core.filters

        biases = Array3D.mapi (fun x y i (w, dw) -> 
          let (bw, bdw) = core.biases.[x,y,i]
          (bw, bdw + dw) ) core.outAct  
      
    }

 

  let forward c (vol:Vol) (training:bool) = 
    let (FullyConnCore core) = c
    let vw = Vol.getWs vol
    let a = 
      Vol.constCreate 1 1 core.outDepth 0.0
      |> Array3D.mapi (fun _ _ i (w,dw) ->
        let wi = Vol.getWs core.filters.[i]
        let bi = Vol.get core.biases 0 0 i 
        let newW = 
          Vol.map2 (*) vw wi 
          |> Vol.flatten
          |> Seq.sum
        (bi + newW, dw)
      )
    
    FullyConnCore { core with 
      inAct = vol
      outAct = a
    }

  let getParamsAndGrads c = 
    let (FullyConnCore core) = c
    let pg = 
      core.filters
      |> List.map (fun f -> { params = Vol.getWs f; grads = Vol.getDWs f; l1DecayMul = core.l1DecayMul; l2DecayMul = core.l2DecayMul })
    pg @ [{ params = Vol.getWs core.biases; grads = Vol.getDWs core.biases; l1DecayMul = 0.0; l2DecayMul = 0.0 }]
    
  let create outDepth l1DecMul l2DecMul inSx inSy inDepth bias = 
    {
      forward = forward
      backward = backward
      getParamsAndGrads = getParamsAndGrads
      data = FullyConnCore {
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
    }
