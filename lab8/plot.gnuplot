set terminal png
set output "plot.png"

set title "Frequency spectrum"
set xlabel "Frequency"
set ylabel "Power"

plot "res.out" with impulses lt rgb "red", "res.out" with points pt 3 lt rgb "red"