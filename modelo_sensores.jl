### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 495a1267-1efe-47b1-9240-18d926945dd0
begin
	using HiGHS
	using JuMP, BenchmarkTools
	using PlotlyJS, LinearAlgebra, Distributions,Random
end

# ╔═╡ 16ea007e-fbeb-4efb-bb71-3410714fcbc5
md"
Considere uma rede 2D com n nós beacon em posições conhecidas xᵢ $\in \mathbb{R}^2, i = 1, ..., n$
"

# ╔═╡ e1fd6dcc-6628-425b-b89f-e963ee0f92a4
md"
Assumimos uma coordenada $x^k = (x,y)$ como uma posição desconhecida inicial 
"

# ╔═╡ f2909155-0391-4ff9-a677-ce0d1e9b01b0
md"
O nó alvo $x^k$ vai transmitir um sinal no instante t₀ e os nós beacon irão receber o sinal no instante tᵢ conforme a expressão abaixo 
"

# ╔═╡ a13aef77-305f-470d-9d32-ddc68e954dfd
md"
$tᵢ = t₀ + \frac{dᵢ}{v} + \omega ᵢ$
"

# ╔═╡ 5ba5c152-2d76-466d-8cb1-483b067784d7
md"
em que $dᵢ = \sqrt[2]{(xᵢ-x)^2+(yᵢ-y)^2}$ conforme a distancia euclidiana entre o beacon i e o ponto x, sendo v a velocidade de propagação do sinal
"

# ╔═╡ 02b8ba3e-f420-4e3c-9197-4d9d0ab614fe
md"

 $\omega ᵢ \approx n(0, \frac{\sigmaᵢ ^2}{v^2})$
"

# ╔═╡ be29c3d3-671a-4f53-9f3d-23daa1fec43f
md"
Supondo que t₀ é um parâmetro desconhecido fixo durante o processo. Para evitar a influência deste parâmetro, subtraimos todas as medições tᵢ da última medição.
Consequentemente, pode ser expresso como
"

# ╔═╡ b73d7d96-f3da-47fd-aea6-6e5f6439aad3
md"
$tᵢ - tₙ = (t₀ + \frac{dᵢ}{v} + \omega ᵢ) - (t₀ + \frac{dₙ}{v} + \omega ₙ)$
"

# ╔═╡ 9de256c0-ec7b-4561-8120-c3e1c57620c4
md"
$tᵢ - tₙ = \frac{dᵢ+dₙ}{v} + \omega ᵢ - \omega ₙ$
"

# ╔═╡ 5df859d6-f2fd-44cd-aacd-27512dad7984
md"
$v(tᵢ - tₙ) = dᵢ+dₙ + v(\omega ᵢ - \omega ₙ)$
"

# ╔═╡ 3decad43-e48f-4e97-a75e-0aab33fc60e7
md"
incluindo variaveis para cada parte da equação 
"

# ╔═╡ 5bb3b101-c661-4f82-8566-c65511025522
md"
$rᵢ = v(tᵢ - tₙ)$
"

# ╔═╡ e9b97d1c-7d52-4663-b15f-243388a5877b
md"
$d_{i,n} = dᵢ+dₙ$
"

# ╔═╡ 4399732e-bdd2-4679-9db5-5e6cb06f3c59
md"
$X_{i} = v(\omega ᵢ - \omega ₙ)$
"

# ╔═╡ 620ee219-31df-4d87-96de-635df08f8dbc
md"
obtivemos a seguinte expressão 
"

# ╔═╡ ac0cb710-9c90-4e56-bc28-3f62e248263c
md"
$R = D + X$
"

# ╔═╡ 6c6ce336-7d08-4e16-a860-9bd321c9b28f
md"
sendo R a distância entre os pontos fixos e o ponto alvo (xₖ) considerando os erros,  obtemos
"

# ╔═╡ 1223a590-7d40-45c2-9b15-344fadd6fd45
md"
$R = [r₁,r₂,...,rₙ]$
"

# ╔═╡ fbc0bf3d-5e5a-46a9-956e-b248dd571deb
md"
$D = [d_{1,n},...,d_{n-1,n}]$ 
"

# ╔═╡ 96534b74-0d11-4690-9bfb-b6c861ef4ec7
md"
$X = [X_{1},...,X_{n-1}]$
"

# ╔═╡ d4453ef4-2812-4609-947c-b6069ebd3a60
md"
consideramos a minimização da norma $\iota_1$ abaixo
"

# ╔═╡ 3b6274fc-0cf6-4599-91af-7b0e416d6084
md"
$\underset{x}{\text{minimize}}\left \| R - D \right \|_1$
"

# ╔═╡ 8acfd714-9fb9-44b5-b565-5519375e3cb7
md"
A norma $\iota_1$ é um problema de otimização não convexo e difícil de resolver, se propõe uma solução eficiente com um método subótimo para resolver este problema. Descrevemos o procedimento de aproximação da norma $\iota_1$ com minimização de resíduos em um problema de otimização convexo, usando um vetor fictício s.
"

# ╔═╡ 4689f6d0-c75b-4079-a770-b644719bc408
md"
$s = [s_1,...,s_{n-1}]^T$
"

# ╔═╡ 9070dc74-242c-434a-bb91-9f71356f9b7a
md"
$\underset{x,s}{\text{minimize}} \sum_{n-1}^{i=1} s_i$
"

# ╔═╡ 2f5fe746-d647-49e9-a279-9ff15608ba9a
md"
sujeito a $s_i = r_i + \left \| x - x_n \right \|_2 - \left \| x - x_i \right \|_2$
"

# ╔═╡ 6740838d-5375-4dde-b466-302ee8917cab
md"
O problema em questão é não convexo e não linear. No caso de relações sinal-ruído suficientemente altas, a restrição acima pode ser aproximado usando a expansão em série de Taylor de primeira ordem, de modo que podemos estudar como a função f se comporta em torno de um ponto fixo $x^k$
"

# ╔═╡ 3dd2f886-cadc-4977-9fb7-6775a91f4482
md"
```math
\begin{matrix} \min & \displaystyle \sum_{i=1}^{n-1} s_i\\
	            \text{s. a.}     & A_ix + C_i -s_i = 0
\end{matrix}
```
"

# ╔═╡ bd88545f-9cc6-46bc-b81e-eaf394e3246b
md"""
conforme a expansão de taylor de primeira ordem:

"""

# ╔═╡ 11b437a7-1d58-42d4-bf60-0959fdf26cb3
md"
$f(x)=f(x_{0})+\frac{f'(x_{0})}{1!}(x-x_{0})$
"

# ╔═╡ 360738ab-7a14-4196-b89c-d44317c56dac
md"""
Sendo assim temos $s_i(x_k)= r_i + \left \| x_k - x_n \right \|_2 - \left \| x_k - x_i \right \|_2$
"""

# ╔═╡ 3859a6eb-fc47-4113-bdf0-1dc56b3447ba
md"""
$$s_i(x_k)' = \frac{x_{k}-x_{n}}{\left \| x_{k} - x_{n} \right \|_2 }-\frac{x_{k}-x_{i}}{\left \| x_{k} - x_{i} \right \|_2 }$$
"""

# ╔═╡ bfcef6d4-11fb-4ff2-84c9-6a82b961d71d
md"""
substituindo na expansão de taylor:
"""

# ╔═╡ 6af33ac2-6195-401e-9ccf-2ef620d2ad63
md"""
$$f(x)=r_i + \left \| x_{k} - x_n \right \|_2 - \left \| x_{k} - x_i \right \|_2 +(\frac{x_{k}-x_{n}}{\left \| x_{k} - x_{n} \right \|_2 }-\frac{x_{k}-x_{i}}{\left \| x_{k} - x_{i} \right \|_2 })(x-x_{k})$$
"""

# ╔═╡ f12fa68a-7169-4ca5-b437-8b3312a1bc74
md"""
$$f(x)=r_i + \left \| x_{k} - x_n \right \|_2 - \left \| x_{k} - x_i \right \|_2 +x\frac{x_{k}-x_{n}}{\left \| x_{k} - x_{n} \right \|_2 }-x\frac{x_{k}-x_{i}}{\left \| x_{k} - x_{i} \right \|_2 }$$
"""

# ╔═╡ 064a0184-346d-473f-acde-dfd14269e12a
md"
$$-x_{k}\frac{x_{k}-x_{n}}{\left \| x_{k} - x_{n} \right \|_2 }+x_{k}\frac{x_{k}-x_{i}}{\left \| x_{k} - x_{i} \right \|_2 }$$
"

# ╔═╡ 25beefb9-502a-43b7-9f2f-791d300f6805
md"
incluindo variaveis para cada parte da equação 
"

# ╔═╡ 84fe2f38-15d3-4f13-a7e2-c6e6d1a477bd
md"
$$A_i = (\frac{x_{k}-x_{n}}{\left \| x_{k} - x_{n} \right \|_2 }-\frac{x_{k}-x_{i}}{\left \| x_{k} - x_{i} \right \|_2 })$$
"

# ╔═╡ 436080f8-a04a-4099-a84a-9e62db7a7682
md"
$$C_i = r_i + \left \| x_{k} - x_n \right \|_2 - \left \| x_{k} - x_i \right \|_2-x_{k}\frac{x_{k}-x_{n}}{\left \| x_{k} - x_{n} \right \|_2 }+x_{k}\frac{x_{k}-x_{i}}{\left \| x_{k} - x_{i} \right \|_2 }$$
"

# ╔═╡ cc2b83e1-f73f-4b12-9838-1f0814608ae8
md"""
Relaxando a minimização anterior obtivemos:
"""

# ╔═╡ cd85ee94-b233-42ce-b95d-87fa83c4cc18
md"""
$$A_{i}x+C_{i}-S_{i}\leq 0$$
"""

# ╔═╡ 5357d93c-7e33-4b51-a258-2c02d7ecfa55
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

# ╔═╡ 98edb114-343b-4d36-95aa-a19946a43fd4
function solucao_LP(A, C, n)
	IC = Model(HiGHS.Optimizer)
 	@variable(IC,x[1:2]>=0)
 	@variable(IC,sᵢ[1:n-1]>=0)
	@objective(IC, Min,sum(sᵢ))
	@constraint(IC,(A*x)+C-sᵢ .== 0)
	print(IC)
 	optimize!(IC)
 	println("Termination status: $(termination_status(IC))")
 	if termination_status(IC) == MOI.OPTIMAL
		println("Optimal objective value: $(objective_value(IC))")
 		println("x: ",value.(x))
	else
 		@constraint(IC,(A*x)+C-sᵢ .<= 0)
		print(IC)
 		optimize!(IC)
 		println("Termination status: $(termination_status(IC))")
		if termination_status(IC) == MOI.OPTIMAL
			println("Optimal objective value: $(objective_value(IC))")
 			println("x: ",value.(x))
	    	println("s: ",value.(sᵢ))
		else
			println("No optimal solution available")
		end
 	end
end

# ╔═╡ 7b8c808b-113b-46e0-819b-7bb7db2c970a
let
function Input(prompt)
    print(prompt)
    readline()
end

n = Input("valores de X, digitar [[12.,12],[12.,12]]\n")
println("Your name is $n.")
end

# ╔═╡ c4595db0-c41d-4671-aee3-8f7f95b10aae
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

# ╔═╡ fa94b397-9be9-434d-b023-d54a898b13c3
##http://www.cnmac.org.br/novo/index.php/CNMAC/conteudo/2022/53/99

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.3.1"
Distributions = "~0.25.76"
HiGHS = "~1.2.0"
JuMP = "~1.3.1"
PlotlyJS = "~0.18.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "07287f7854c3f75419346d4f8ed3c89a8c09f1ad"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[deps.Blink]]
deps = ["Base64", "BinDeps", "Distributed", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Reexport", "Sockets", "WebIO", "WebSockets"]
git-tree-sha1 = "08d0b679fd7caa49e2bca9214b131289e19808c0"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "e08915633fcb3ea83bf9d6126292e5bc5c739922"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.13.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "c5b6685d53f933c11404a3ae9822afe30d522494"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.12.2"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "04db820ebcfc1e053bd8cbb8d8bccf0ff3ead3f7"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.76"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "9a0472ec2f5409db243160a8b030f94c380167a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.6"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HiGHS]]
deps = ["HiGHS_jll", "MathOptInterface", "SparseArrays"]
git-tree-sha1 = "d40a9e8db6438481915261a378fc2c8ca70bb63a"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.2.0"

[[deps.HiGHS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3e24275666fcc2e24d1a58a9f02acd9d2e23d3a"
uuid = "8fd58aa0-07eb-5a78-9b36-339c94fd15ea"
version = "1.3.0+0"

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays"]
git-tree-sha1 = "8c0aacbcb0530d6fdc2650fe8cd312e7da452dbc"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.3.1"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "09c6964bf4bca818867494739a9387c0c9cf4e2c"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "1e566ae913a57d0062ff1af54d2697b9344b99cd"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.14"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "aa532179d4a643d4bd9f328589ca01fa20a0d197"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.1.0"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "Pkg", "Sockets", "WebSockets"]
git-tree-sha1 = "82dfb2cead9895e10ee1b0ca37a01088456c4364"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "0.7.6"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "b64719e8b4504983c7fca6cc9db3ebc8acc2a4d6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "7452869933cd5af22f59557390674e8679ab2338"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.10"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "ffc098086f35909741f71ce21d03dadf0d2bfa76"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.11"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "e4bdc63f5c6d62e80eb1c0043fcc0360d5950ff7"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.10"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "55ea1b43214edb1f6a228105a219c6e84f1f5533"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.19"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "f91a602e25fe6b89afc93cf02a4ae18ee9384ce3"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.5.9"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═495a1267-1efe-47b1-9240-18d926945dd0
# ╟─16ea007e-fbeb-4efb-bb71-3410714fcbc5
# ╟─e1fd6dcc-6628-425b-b89f-e963ee0f92a4
# ╟─f2909155-0391-4ff9-a677-ce0d1e9b01b0
# ╠═a13aef77-305f-470d-9d32-ddc68e954dfd
# ╟─5ba5c152-2d76-466d-8cb1-483b067784d7
# ╟─02b8ba3e-f420-4e3c-9197-4d9d0ab614fe
# ╟─be29c3d3-671a-4f53-9f3d-23daa1fec43f
# ╟─b73d7d96-f3da-47fd-aea6-6e5f6439aad3
# ╟─9de256c0-ec7b-4561-8120-c3e1c57620c4
# ╟─5df859d6-f2fd-44cd-aacd-27512dad7984
# ╟─3decad43-e48f-4e97-a75e-0aab33fc60e7
# ╟─5bb3b101-c661-4f82-8566-c65511025522
# ╟─e9b97d1c-7d52-4663-b15f-243388a5877b
# ╟─4399732e-bdd2-4679-9db5-5e6cb06f3c59
# ╟─620ee219-31df-4d87-96de-635df08f8dbc
# ╟─ac0cb710-9c90-4e56-bc28-3f62e248263c
# ╟─6c6ce336-7d08-4e16-a860-9bd321c9b28f
# ╟─1223a590-7d40-45c2-9b15-344fadd6fd45
# ╟─fbc0bf3d-5e5a-46a9-956e-b248dd571deb
# ╟─96534b74-0d11-4690-9bfb-b6c861ef4ec7
# ╟─d4453ef4-2812-4609-947c-b6069ebd3a60
# ╠═3b6274fc-0cf6-4599-91af-7b0e416d6084
# ╟─8acfd714-9fb9-44b5-b565-5519375e3cb7
# ╟─4689f6d0-c75b-4079-a770-b644719bc408
# ╟─9070dc74-242c-434a-bb91-9f71356f9b7a
# ╟─2f5fe746-d647-49e9-a279-9ff15608ba9a
# ╟─6740838d-5375-4dde-b466-302ee8917cab
# ╟─3dd2f886-cadc-4977-9fb7-6775a91f4482
# ╟─bd88545f-9cc6-46bc-b81e-eaf394e3246b
# ╟─11b437a7-1d58-42d4-bf60-0959fdf26cb3
# ╟─360738ab-7a14-4196-b89c-d44317c56dac
# ╟─3859a6eb-fc47-4113-bdf0-1dc56b3447ba
# ╟─bfcef6d4-11fb-4ff2-84c9-6a82b961d71d
# ╟─6af33ac2-6195-401e-9ccf-2ef620d2ad63
# ╟─f12fa68a-7169-4ca5-b437-8b3312a1bc74
# ╟─064a0184-346d-473f-acde-dfd14269e12a
# ╟─25beefb9-502a-43b7-9f2f-791d300f6805
# ╟─84fe2f38-15d3-4f13-a7e2-c6e6d1a477bd
# ╟─436080f8-a04a-4099-a84a-9e62db7a7682
# ╟─cc2b83e1-f73f-4b12-9838-1f0814608ae8
# ╟─cd85ee94-b233-42ce-b95d-87fa83c4cc18
# ╠═5357d93c-7e33-4b51-a258-2c02d7ecfa55
# ╠═98edb114-343b-4d36-95aa-a19946a43fd4
# ╠═7b8c808b-113b-46e0-819b-7bb7db2c970a
# ╠═c4595db0-c41d-4671-aee3-8f7f95b10aae
# ╠═fa94b397-9be9-434d-b023-d54a898b13c3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
