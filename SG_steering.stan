data{

	int N;
	
	vector[N] near_dot;
	vector[N] far_dot;
	vector[N] near;
	real<lower = 0> dt;
	vector[N] heading_dot;
	
}

parameters{

	real k_n;
	real k_i_n;
	real k_f;	
	
	real<lower = 0> sigma;
}


model{

	heading_dot ~ normal(k_i_n * near * dt + k_n * near_dot + k_f * far_dot, sigma);

}

generated quantities{

	real log_lik[N];
	
	for (n in 1:N){
		log_lik[n] = normal_lpdf(heading_dot[n] | k_i_n * near * dt + k_n * near_dot + k_f * far_dot, sigma);
	}

}
