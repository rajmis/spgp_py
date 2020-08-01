from minimize import minimize
import time
# import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objs as go

from SPGP_util import *
from numpy import *

grid_max_x = 200
grid_max_y = 200
grid_min = 1
M = 100
dim = 2
funcEval = 500


data_temp = loadtxt('data_temp').reshape(-1,1)
x = loadtxt('sense_loc_x',dtype=float,delimiter=',')
y = loadtxt('sense_loc_y',dtype=float,delimiter=',')
sense_loc = array([x.T,y.T]).T
print(str(data_temp[100]) + "--> " + str(x[100]) + ", " + str(y[100]))
## Generating rando
x = loadtxt('xb_init_x',dtype=float,delimiter=',')
y = loadtxt('xb_init_y',dtype=float,delimiter=',')
# xb_init = (grid_max_y-grid_min)*random.rand(M,2) + grid_min
xb_init = array([x.T,y.T]).T
# hyp_init = -2*log(max(x)-min(x))/2
# hyp_init = append(hyp_init,-2*log(max(y)-min(y))/2)

# hyp_init = append(hyp_init,log(var(data_temp)))
# hyp_init = append(hyp_init,log(var(data_temp)/4))
hyp_init = loadtxt('hyp_init',dtype=float,delimiter=',')
hyp_init = array(hyp_init.T)
w_init = append(reshape(xb_init,[M*dim,-1],order='F'),reshape(hyp_init,[len(hyp_init),-1],order='F'))
w_init = reshape(w_init,[len(w_init),-1])
y_mean = mean(data_temp)
y0 = data_temp - y_mean
# start_time = time.time()
# print sense_loc.shape
# print y0
start_time = time.time()
# for i in range(1,15000):
# print(spgp_lik(w_init,y0,sense_loc,M))
[w,f,i] = minimize(w_init,spgp_lik,args=[y0,sense_loc,M],maxnumfuneval=-funcEval,verbose=True)
xb = reshape(w[0:-dim-2],(M,dim),order='F')
hyp = reshape(w[-dim-2:],(dim+2,1),order='F')
# print(w_init)
# print(w)
# print(hyp_init)
# print(hyp)
# print(exp(hyp))

# savetxt('xb_iter_x',xb[:,0],delimiter=',')
# savetxt('xb_iter_y',xb[:,1],delimiter=',')
# savetxt('hyp_learn',hyp,delimiter=',')
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

# print(w)
# res = minimize(spgp_lik_nograd,w_init,method='BFGS',args=(y0,sense_loc,M),options={'maxiter':10,'disp':True})

# hyp_init = -2*np.log((np.max(sense)))

# x = array([x.T,x.T+1]).T;
# # w = array([0.3944,2.6315,4.0440,1.3944,3.6315,5.0440,0.6733,0.6733,-0.7497,-6.3823]).reshape(-1,1);
# w_init = loadtxt('w_init').reshape(-1,1);
# y = array(loadtxt('train_outputs_new')).reshape(-1,1);
# M=3;
# y0 = y - mean(y);
# # print spgp_lik
# [w,f,i] = minimize(w_init,spgp_lik,args=[y0,x,M],maxnumfuneval=-15000,verbose=False);

# a = spgp_lik(w_init,y0,sense_loc,M)
t= time.time() - start_time
# print a
print(t)
field_robo = mu.reshape(200,200)
field_robo_var = s2.reshape(200,200)

savetxt('xb',xb)
savetxt('hyp',hyp)
savetxt('field_robo_var',field_robo_var)

# CS = plt.contourf(field_robo,cmap='jet')
# cbar = plt.colorbar(CS)

# plt.figure()
# CS2 = plt.contourf(field_robo_var,cmap='jet',vmin=0,vmax=1)
# cbar = plt.colorbar(CS2)
# plt.show()

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
offline.plot(go.Figure(data,layout),auto_open=False,filename='field_robo.html')

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
offline.plot(go.Figure(data,layout),auto_open=False,filename='field_robo_var.html')