function calcula_rᵢ(n,σ,xₖ,X,v)
	ω = similar(σ)
	for (index,σi) in enumerate(σ)
		ω[index] = rand(Normal(0,σi/v))
	end
	ωₙ = pop!(ω)
	ω .-= ωₙ
	ω *= v
	D = ((norm.(X .- [xₖ]))[1:end-1].-(norm.(X .- [xₖ]))[end])
	R = D + ω
	return R
end