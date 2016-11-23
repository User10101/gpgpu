set terminal gif animate delay 100
set output 'upwind.gif'
stats 'plotdata.dat' nooutput
set xr[0:128]
set yr[0:0.5]

do for [i=1:int(STATS_blocks)] {
   plot 'plotdata.dat' index (i-1) with lines
}