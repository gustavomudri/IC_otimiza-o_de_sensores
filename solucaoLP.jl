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