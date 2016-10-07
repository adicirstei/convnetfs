#load "util.fsx"
#load "convnet.fsx"

open Convnet

let randCreate sx sy depth : Vol = 
  let scale = sqrt (1.0 / float (sx * sy * depth))
  Array3D.init sx sy depth (fun x y z -> (Util.randn 0.0 scale, 0.0))

let constCreate sx sy depth c : Vol = Array3D.create<T> sx sy depth (c, 0.0)

let get (vol: Vol) x y d = fst (vol.[x,y,d])

let set (vol: Vol) x y d v = vol.[x,y,d] <- (v, snd vol.[x,y,d]); 

let add (vol: Vol) x y d v = vol.[x,y,d] <- (fst vol.[x,y,d] + v, snd vol.[x,y,d])

let getGrad (vol: Vol) x y d = snd (vol.[x,y,d])

let setGrad (vol: Vol) x y d v = vol.[x,y,d] <- (fst vol.[x,y,d], v); 

let addGrad (vol: Vol) x y d v = vol.[x,y,d] <- (fst vol.[x,y,d], snd vol.[x,y,d] + v);

let cloneAndZero (vol: Vol) : Vol = Array3D.map (fun _ -> (0.0, 0.0)) vol

let clone (vol: Vol) : Vol = Array3D.map (fun (w, dw) -> (w, 0.0)) vol

let addFrom (vol: Vol) (sVol: Vol) = 
  vol
  |> Array3D.iteri (fun x y d (w,dw) -> vol.[x,y,d] <- (w + (fst sVol.[x,y,d]), dw)) 

let addFromScaled (vol: Vol) (sVol: Vol) a = 
  vol
  |> Array3D.iteri (fun x y d (w,dw) -> vol.[x,y,d] <- (w + a * (fst sVol.[x,y,d]), dw)) 

let setConst (vol: Vol) a = 
  vol
  |> Array3D.iteri (fun x y d (w,dw) -> vol.[x,y,d] <- (a, dw)) 

