from spgp.minimize import minimize
import time
import plotly.offline as offline
import plotly.graph_objs as go

from spgp.utilfn import *
from numpy import *

grid_max_x = 200
grid_max_y = 200
grid_min = 1
M = 100
dim = 2


data_temp = loadtxt('data/spatial_regression/data_temp').reshape(-1,1)
x = loadtxt('data/spatial_regression/sense_loc_x',dtype=float,delimiter=',')
y = loadtxt('data/spatial_regression/sense_loc_y',dtype=float,delimiter=',')
sense_loc = array([x.T,y.T]).T
print(str(data_temp[100]) + "--> " + str(x[100]) + ", " + str(y[100]))
x = loadtxt('data/spatial_regression/xb_init_x',dtype=float,delimiter=',')
y = loadtxt('data/spatial_regression/xb_init_y',dtype=float,delimiter=',')

xb_init = array([x.T,y.T]).T
hyp_init = loadtxt('data/spatial_regression/hyp_init',dtype=float,delimiter=',')
hyp_init = array(hyp_init.T)
w_init = append(reshape(xb_init,[M*dim,-1],order='F'),reshape(hyp_init,[len(hyp_init),-1],order='F'))
w_init = reshape(w_init,[len(w_init),-1])
y_mean = mean(data_temp)
y0 = data_temp - y_mean
start_time = time.time()
[w,f,i] = minimize(w_init,spgp_lik,args=[y0,sense_loc,M],maxnumfuneval=-500,verbose=True)
xb = reshape(w[0:-dim-2],(M,dim),order='F')
hyp = reshape(w[-dim-2:],(dim+2,1),order='F')

x = linspace(0,200,200)
y = linspace(0,200,200)
[X,Y] = meshgrid(x,y)
X = X.reshape(200*200)
Y = Y.reshape(200*200)
grid_world = array([X,Y]).T
[mu,s2] = spgp_pred(y0,sense_loc,xb,grid_world,hyp)
mu = mu + y_mean
s2 = s2 + exp(hyp[-1])
print(exp(hyp[-2]))
print(s2.min())
print(mu.max())

t= time.time() - start_time
print(t)
field_robo = mu.reshape(200,200)
field_robo_var = s2.reshape(200,200)

trace1 = go.Contour(
		z=field_robo,
		x=x,
		y=y,
		colorscale='Jet',
		colorbar=dict(
			title='Temperature (K)',
			titleside='right',
			)
	)

trace2 = go.Scatter(
		x = sense_loc[:,0],
		y = sense_loc[:,1],
		mode='markers',
		marker = dict(
    	size=10,
    	color = 'rgba(255, 255, 255, .5)'
    	)
	)

layout = go.Layout(
	title = 'Predicted Field',
	xaxis=dict(
		title='<--x(m)-->'
	),
	yaxis=dict(
		title='<--y(m)-->'
	)
)
data = [trace1,trace2]
offline.plot(go.Figure(data,layout),auto_open=False,filename='output/mean_2Dfield.html')

data = [
	go.Contour(
		z=field_robo_var,
		x=x,
		y=y,
		colorscale='Jet',
		zmin=0,
		zmax=1,
	)
]
layout = go.Layout(
	title = 'Predicted Variance',
	xaxis=dict(
		title='<--x(m)-->'
	),
	yaxis=dict(
		title='<--y(m)-->'
	)
)
offline.plot(go.Figure(data,layout),auto_open=False,filename='output/variance_2Dfield.html')