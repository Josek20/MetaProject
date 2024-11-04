
theory = @theory a b c d x y begin
    a::Number + b::Number => a + b
    a::Number - b::Number => a - b
    a::Number * b::Number => a * b
    # a::Number / b::Number => b != 0 ? a / b : :($a / $b)
    # add.rc
    a + b --> b + a
    a + (b + c) --> (a + b) + c
    (a + b) + c --> a + (b + c)
    (a + b) - c --> a + (b - c)
    a + (b - c) --> (a + b) - c
    (a - b) - c --> a - (b + c) 
    a - (b + c) --> (a - b) - c
    (a - b) + c --> a - (b - c)
    a - (b - c) --> (a - b) + c
    # (a + b) - c --> a + (b - c)
    # a + (b - c) --> (a + b) - c 
    a + 0 --> a
    a * (b + c) --> a*b + a*c
    a*b + a*c --> a * (b + c)
    (a / b) + c --> (a + (c * b)) / b
    (a + (c * b)) / b --> (a / b) + c
    x / 2 + x % 2 --> (x + 1) / 2
    x * a + y * b --> ((a / b) * x + y) * b
    
    # and.rc
    a && b --> b && a
    a && (b && c) --> (a && b) && c
    1 && a --> a
    a && 1 --> a
    a && a --> a
    a && !a --> 0
    !a && a --> 0

    (a == c::Number) && (a == x::Number) => c != x ? 0 : :($a == $c)
    !a::Number && b::Number => a != b ? 0 : :($a)
    (a < y) && (a < b) --> a < min(y, b)
    a < min(y, b) --> (a < y) && (a < b)
    (a <= y) && (a <= b) --> a <= min(y, b)
    a <= min(y, b) --> (a <= y) && (a <= b)
    (a > y) && (a > b) --> x > max(y, b)
    a > max(y, b) --> (a > y) && (a > b)
    
    (a >= y) && (a >= b) --> a >= max(y, b)
    a >= max(y, b) --> (a >= y) && (a >= b)
    
    (a::Number > b) && (c::Number < b) => a < c ? 0 : :($a > $b) && :($c < $b)
    (a::Number >= b) && (c::Number <= b) => a < c ? 0 : :($a >= $b) && :($c <= $b)
    (a::Number >= b) && (c::Number < b) => a <= c ? 0 : :($a >= $b) && :($c < $b)
    
    a && (b || c) --> (a && b) || (a && c)
    a || (b && c) --> (a || b) && (a || c)
    b || (b && c) --> b

    # div.rs
    0 / a --> 0
    a / a --> 1
    (-1 * a) / b --> a / (-1 * b)
    a / (-1 * b) --> (-1 * a) / b
    -1 * (a / b) --> (-1 * a) / b
    (-1 * a) / b --> -1 * (a / b)

    (a * b) / c --> a / (c / b)
    a / (c / b) --> (a * b) / c
    (a / b) * c --> a / (b / c)
    a / (b / c) --> (a / b) * c
    # (a * b) / c --> a * (b / c) ?
    (a + b) / c --> (a / c) + (b / c)
    ((a * b) + c) / d --> ((a * b) / d) + (c / d)

    # eq.rs
    x == y --> y == x
    x == y => y != 0 ? :(($x - $y) == 0) : :($x == $y)
    x + y == a --> x == a - y
    x == x --> 1
    x*y == 0 --> (x == 0) || (y == 0)
    max(x,y) == y --> x <= y
    min(x,y) == y --> y <= x
    y <= x --> min(x,y) == y

    # ineq.rs
    x != y --> !(x == y)

    # lt.rs
    x > y --> y < x
    # x < y --> (-1 * y) < (-1 * x)
    a < a --> 0
    a <= a --> 1
    a + b < c --> a < c - b

    a - b < a --> 0 < b
    0 < a::Number => 0 < a ? 1 : 0
    a::Number < 0 => a < 0 ? 1 : 0
    min(a , b) <= a --> b <= a
    min(a, b) <= min(a , c) --> b <= min(a, c)
    max(a, b) <= max(a , c) --> max(a ,b) <= c

    # max.rs
    # max(a, b) --> (-1 * min(-1 * a, -1 * b))
    
    # min.rs
    min(a, b) --> min(b, a)
    min(min(a, b), c) --> min(a, min(b, c))
    min(a,a) --> a
    min(max(a, b), a) --> a
    min(max(a, b), max(a, c)) --> max(min(b, c), a)
    min(max(min(a,b), c), b) --> min(max(a,c), b)
    min(a + b, c) --> min(b, c - a) + a
    min(a, b) + c --> min(a + c, b + c)
    
    min(a + c, b + c) --> min(a, b) + c
    min(c + a, b + c) --> min(a, b) + c
    min(a + c, b + c) --> min(a, b) + c
    min(c + a, c + b) --> min(a, b) + c

    min(a, a + b::Number) => b > 0 ? :($a) : :($a + $b)
    min(a ,b) * c::Number => c > 0 ? :(min($a * $c, $b * $c)) : :(max($a * $c, $b * $c)) 
    min(a * c::Number, b * c::Number) => c > 0 ? :(min($a ,$b) * $c) : :(max($a, $b) * $c)
    min(a, b) / c::Number => c > 0 ? :(min($a / $c, $b / $c)) : :(max($a/$c,$b/$c))
    min(a / c::Number, b / c::Number) => c>0 ? :(min($a, $b) / $c) : :(max($a,$b) / $c)
    max(a , b) / c::Number => c < 0 ? :(max($a / $c , $b / $c)) : :(min($a / $c, $b / $c))
    max(a / c::Number, b / c::Number) => c < 0 ? :(max($a, $b) / $c) : :(min($a, $b) / $c)
    min(max(a,b::Number), c::Number) => c <= b ? :($c) : :(min(max($a,$b),$c))
    # min((a / b::Number) * b::Number , a) => b > 0 ? :(($a / $b) * $b) : :($)
    min(a % b::Number, c::Number) => c >= b - 1 ? :($a % $b) : :(min($a % $b, $c))
    min(a % b::Number, c::Number) => c <= 1 - b ? :($c) : :(min($a % $b, $c))

    min(max(a, b::Number), c::Number) => b <= c ? :(max(min($a, $c), $b)) : :(min(max($a, $b), $c))
    max(min(a, c::Number), b::Number) => b <= c ? :(min(max($a, $b), $c)) : :(max(min($a, $c), $b))
    min(a , b::Number) <= c::Number --> a <= c || b <= c
    max(a , b::Number) <= c::Number --> a <= c && b <= c
    c::Number <= max(a , b::Number) --> c <= a || c <= b
    c::Number <= min(a , b::Number) --> c <= a && c <= b
    min(a * b::Number, c::Number) => c != 0 && b % c == 0 && b > 0 ? :(min($a, $b / $c) * $c) : :(min($a * $b, $c))
    min(a * b::Number, d * c::Number) => c != 0 && b % c == 0 && b > 0 ? :(min($a, $d * ($c/$b))*$b) : :(min($a * $b, $d * $c))
    min(a * b::Number, c::Number) => c != 0 && b % c == 0 && b < 0 ? :(max($a, $b / $c) * $c) : :(min($a * $b, $c))
    min(a * b::Number, d * c::Number) => c != 0 && b % c == 0 && b < 0 ? :(max($a, $d * ($c/$b))*$b) : :(min($a * $b, $d * $c))
    max(a * c::Number, b * c::Number) => c < 0 ? :(min($a, $b) * $c) : :(max($a, $b) * $c)

    # modulo.rs
    a % 0 --> 0
    a % a --> 0
    a % 1 --> 0

    # a % b::Number --> b > 0 ? :(($a + $b) % $b) : :($a % $b)
    (a * -1) % b --> -1 * (a % b)
    -1 * (a % b) --> (a * -1) % b
    (a - b) % 2 --> (a + b) % 2

    ((a * b::Number) + d) % c::Number => c != 0 && b % c == 0 ? :($b % $c) : :((($a * $b) + $d) % $c)
    (b::Number * a) % c::Number => c != 0 && b % c == 0 ? 0 : :(($b * $a) % $c)
    
    # mul.rs
    a * b --> b * a
    # a * b::Number --> b * a
    a * (b * c) --> (a * b) * c
    a * 0 --> 0
    0 * a --> 0
    a * 1 --> a
    1 * a --> a
    # (a / b) * b --> (a - (a % b))
    max(a,b) * min(a, b) --> a * b
    min(a,b) * max(a, b) --> a * b
    (a * b) / b --> a
    (b * a) / b --> a

    # not.rs
    x <= y --> !(y < x)
    !(y < x) --> x <= y
    x >= y --> !(x < y)
    !(x == y) --> x != y
    !(!x) --> x

    # or.rs
    x || y --> !((!x) && (!y))
    y || x --> x || y

    # sub.rs
    # a - b --> a + (-1 * b)
    
    # my rules
    a || 1 --> 1
    1 || a --> 1
    a::Number <= b::Number => a<=b ? 1 : 0
    a <= b - c --> a + c <= b
    a + c <= b --> a <= b - c
    #a <= b - c --> a - b <= -c
    a <= b + c --> a - c <= b
    a - c <= b --> a <= b + c
    a <= b + c --> a - b <= c
    a - b <= c --> a <= b + c
    a - a --> 0
    min(a::Number, b::Number) => a >= b ? b : a 
    max(a::Number, b::Number) => a >= b ? a : b
    # a + b::Number <= min(a + c::Number, d) =>
    a <= min(b,c) --> a<=b && a<=c
    min(b,c) <= a --> b<=a || c<=a
    a <= max(b,c) --> a<=b || a<=c
    max(b,c) <= a --> b<=a || c<=a
    a<=b && c<=a --> c <= b
    b<=a && a<=c --> b <= c
    a + b - c --> a - c + b
end