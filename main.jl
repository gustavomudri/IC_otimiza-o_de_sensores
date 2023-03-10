let

	#######MAIN###########
	Random.seed!(10)
	X = [[0.,0],[100,100.],[0,100.],[100,0.]]
	n = length(X)
	xₖ = [111,99]
	v = 20
	σ = Float64[.5,2,2.0,1]
	R = calcula_rᵢ(n,σ,xₖ,X,v)
	Aᵢ(xᵢ) = ((xₖ - X[end])/norm(xₖ - X[end]))  - ((xₖ - xᵢ)/norm(xₖ - xᵢ))
	Cᵢ(xᵢ) = norm(xₖ - X[end]) -  norm(xₖ - xᵢ) + dot(Aᵢ(xᵢ),-xₖ)
	C = R +  Cᵢ.(X[1:end-1])
	A = Aᵢ.(X[1:end-1])
	A = permutedims(hcat(A...))
	solucao_LP(A,C,n)

end