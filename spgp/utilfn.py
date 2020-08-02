from numpy import reshape,exp,tile,sum,log,diag,array,linalg,sqrt,trace,eye,pi,append


"""
This file constains the the utilites functions for learning the SPGP model and using the trained model for estimating the mean and variance.

The spgp_lik is used for calculating the gradient for learning the hyper-parameters of the SPGP model. It assumes the hyper-params are provided as a single weight vector and this function unwraps it.

spgp_pred is used for estimating the mean and variance given a test input.
"""

def spgp_lik(w,y,x,n):
	DEL = 1e-6

	(N,dim) = x.shape
	xb = reshape(w[0:-dim-2],(n,dim),order='F')
	b = reshape(exp(w[-dim-2:-dim]),(dim,1),order='F')
	c = exp(w[-2])
	sig = exp(w[-1])

	tiled_b = tile(sqrt(b).reshape(1,dim),(n,1))
	xb = xb*tiled_b

	tiled_b = tile(sqrt(b).reshape(1,dim),(N,1))
	x = x*tiled_b

	Q = xb.dot(xb.T)
	Q = tile(diag(Q).T,(n,1)).T + tile(diag(Q).T,(n,1)) - 2*Q
	Q = c*exp(-0.5*Q) + DEL*eye(n)

	K = -2*xb.dot(x.T) + tile(sum(x*x,1).T,(n,1)) + tile(sum(xb*xb,1).T,(N,1)).T
	K = c*exp(-0.5*K)

	L = linalg.cholesky(Q)

	invL = linalg.inv(L)
	V = invL.dot(K)
	
	ep = (1 + (c-sum(pow(V,2),axis=0).T)/sig).reshape(-1,1)
	
	K = K/tile(sqrt(ep).T,(n,1))
	
	V = V/tile(sqrt(ep).T,(n,1))
	y = y/sqrt(ep)
	Lm = linalg.cholesky(sig*eye(n) + V.dot(V.T))
	
	invLmV = linalg.inv(Lm).dot(V)
	bet = invLmV.dot(y)

	# Likelihood
	
	fw = sum(log(diag(Lm)))
	fw += (log(sig)*(N-n)/2)[0]
	fw += ((y.T.dot(y)-bet.T.dot(bet))/2/sig)[0]
	fw += sum(log(ep))/2
	fw += 0.5*N*log(2*pi)

	# precomputations
	Lt = L.dot(Lm)
	
	B1 = linalg.inv(Lt.T).dot(invLmV)
	
	b1 = linalg.inv(Lt.T).dot(bet)
	
	invLV = linalg.inv(L.T).dot(V)
	invL = linalg.inv(L)
	invQ = (invL.T).dot(invL)
	invLt = linalg.inv(Lt)
	invA = (invLt.T).dot(invLt)
	
	mu = ((linalg.inv(Lm.T).dot(bet)).T.dot(V)).T
	sumVsq = sum(pow(V,2),axis=0).reshape(-1,1)
	
	bigsum = y*((bet.T).dot(invLmV)).reshape(-1,1)/sig
	bigsum -= sum(invLmV*invLmV,axis=0).reshape(-1,1)/2
	bigsum -= (pow(y,2)+pow(mu,2))/2/sig 
	bigsum += 0.5
	
	TT = invLV.dot(invLV.T*tile(bigsum,(1,n)))

	dfb = array([])
	dfxb = array([])

	for i in range(0,dim):
		
		dnnQ = (tile(xb[:,i].reshape(-1,1),(1,n)) - tile(xb[:,i],(n,1)))*Q
		
		dNnK = (tile(x[:,i],(n,1)) - tile(xb[:,i].reshape(-1,1),(1,N)))*K
		epdot=(-2/sig)[0]*(dNnK)*invLV
		epPmod = -sum(epdot,axis=0).reshape(-1,1)

		
		dfxb_temp = -b1*(dNnK.dot(y-mu)/sig + dnnQ.dot(b1))
		dfxb_temp += sum((invQ-invA*sig)*(dnnQ),axis=1).reshape(-1,1)
		dfxb_temp += epdot.dot(bigsum)
		dfxb_temp -= 2/sig*sum(dnnQ*TT,axis=1).reshape(-1,1)

		dfb_temp = (((y-mu).T*(b1.T.dot(dNnK)))/sig + (epPmod*bigsum).T).dot(x[:,i])
		dNnK = dNnK*B1

		dfxb_temp = dfxb_temp + sum(dNnK,1).reshape(-1,1)
		dfb_temp = dfb_temp - sum(dNnK,0).dot(x[:,i])

		dfxb_temp = dfxb_temp*sqrt(b[i])

		dfb_temp = dfb_temp/sqrt(b[i])
		dfb_temp = dfb_temp + (dfxb_temp.T).dot(xb[:,i])/b[i]
		dfb_temp = dfb_temp*sqrt(b[i])/2

		dfb = append(dfb,dfb_temp)
		dfxb = append(dfxb,dfxb_temp)

	dfb = dfb.reshape(dim,1)
	dfxb = dfxb.reshape(-1,dim,order='F')


	epc = (c/ep-sumVsq-DEL*sum(pow(invLV,2),0).reshape(-1,1))/sig
	# dfc = (n + DEL*trace(invQ - sig*invA) -sig*sum(sum(invA*Q.T,0),0))/2 - mu.T.dot(y-mu)/sig + b1.T.dot(Q-DEL*eye(n)).dot(b1/2) + epc.T.dot(bigsum)
	dfc = (n + DEL*trace(invQ - sig*invA) -sig*sum(sum(invA*Q.T,0),0))[0]/2
	dfc -= mu.T.dot(y-mu)/sig
	dfc += b1.T.dot(Q-DEL*eye(n)).dot(b1/2)
	dfc += epc.T.dot(bigsum)

	dfsig = sum(bigsum/ep,0)

	dfw = reshape(dfxb,(n*dim,1),order='F')
	dfw = append(dfw,dfb,0)
	dfw = append(dfw,dfc,0)
	dfw = append(dfw,[dfsig],0)

	return fw, dfw


def spgp_pred(y,x,xb,xt,hyp,compute_mu=True):
	DEL = 1e-6
	(N,dim) = x.shape
	(Nt,dim) = xt.shape
	(n,dim) = xb.shape

	b = reshape(exp(hyp[-dim-2:-dim]),(dim,1),order='F')
	c = exp(hyp[-2])
	sig = exp(hyp[-1])


	tiled_b = tile(sqrt(b).reshape(1,dim),(n,1))
	xb = xb*tiled_b
	
	tiled_b = tile(sqrt(b).reshape(1,dim),(N,1))
	x = x*tiled_b

	tiled_b = tile(sqrt(b).reshape(1,dim),(Nt,1))
	xt = xt*tiled_b

	Q = xb.dot(xb.T)
	
	Q = tile(diag(Q),(n,1)).T + tile(diag(Q),(n,1)) - 2*Q
	Q = c*exp(-0.5*Q) + DEL*eye(n)
	L = linalg.cholesky(Q)

	K = -2*xb.dot(x.T) + tile(sum(x*x,1).T,(n,1)) + tile(sum(xb*xb,1).T,(N,1)).T
	K = c*exp(-0.5*K)

	invL = linalg.inv(L)
	V = invL.dot(K)
	
	
	tiled_diag_c = tile(c, (1,N))
	ep = (1 + (tiled_diag_c - sum(pow(V,2),axis=0))/sig).reshape(-1,1)
	V = V/tile(sqrt(ep).T,(n,1))
	

	Lm = linalg.cholesky(sig*eye(n) + V.dot(V.T))
	invLm = linalg.inv(Lm)

	K_xt = -2*xb.dot(xt.T) + tile(sum(xt*xt,1).T,(n,1)) + tile(sum(xb*xb,1).T,(Nt,1)).T
	K_xt = c*exp(-0.5*K_xt)

	lst = invL.dot(K_xt)
	lmst = invLm.dot(lst)

	
	s2 = tile(c, (1,Nt))
	s2 -= sum(pow(lst,2),axis=0)
	s2 += sig*sum(pow(lmst,2),axis=0)

	if(not compute_mu):
		return s2

	y = y/sqrt(ep)
	bet = invLm.dot(V.dot(y))
	mu = (bet.T.dot(lmst)).T

	return mu, s2