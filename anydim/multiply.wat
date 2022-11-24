(module
  (import "js" "mem" (memory 1))
  (func (export "multiply")
      (param $dv i32) (param $dm i32)
      (param $mat i32) (param $in i32) (param $out i32)
      (local $i i32) (local $j i32)
      (local $sum f32) (local $inptr i32)
    i32.const 0
    local.set $i
    (loop $i_loop
      i32.const 0
      local.set $j

      f32.const 0
      local.set $sum

      local.get $in
      local.set $inptr

      (loop $j_loop
        local.get $sum
        local.get $mat
        f32.load

        local.get $inptr
        f32.load

        f32.mul

        f32.add
        local.set $sum

        local.get $mat
        i32.const 4
        i32.add
        local.set $mat

        local.get $inptr
        i32.const 4
        i32.add
        local.set $inptr

        local.get $j
        i32.const 1
        i32.add
        local.set $j

        local.get $j
        local.get $dm
        i32.lt_s
        br_if $j_loop
      )

      local.get $out
      local.get $sum
      f32.store

      local.get $out
      i32.const 4
      i32.add
      local.set $out

      local.get $i
      i32.const 1
      i32.add
      local.set $i

      local.get $i
      local.get $dv
      i32.lt_s
      br_if $i_loop
    )
  )
)
