module Vol.Util

#load "vol.fsx"

open System.Drawing

let imageToVol (img:Bitmap) = 
  Array3D.init img.Width img.Height 4 
    (fun x y d ->
      let c:Color = img.GetPixel(x, y)
      match d with
      | 0 -> c.R
      | 1 -> c.G
      | 2 -> c.B
      | _ -> c.A
    )