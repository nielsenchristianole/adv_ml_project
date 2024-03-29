latentdim = 32
device = cuda

train-vae-gaus:
	python3 vae/vae.py train --prior gaus --model models/vae_gaus_model_ld_$(latentdim).pt --latent-dim $(latentdim) --device $(device)

train-vae-mog:
	python3 vae/vae.py train --prior mog  --model models/vae_mog_model_ld_$(latentdim).pt --latent-dim $(latentdim) --device $(device)

train-vae-vampprior:
	python3 vae/vae.py train --prior vampprior --model models/vae_vampprior_model_ld_$(latentdim).pt --latent-dim $(latentdim) --device $(device)

train-vae-flow:
	python3 vae/vae.py train --prior flow --model models/vae_flow_model_ld_$(latentdim).pt --latent-dim $(latentdim) --device $(device)

train: train-vae-gaus train-vae-mog train-vae-vampprior train-vae-flow

vis-vae-gaus:
	python3 vae/vae.py vis --model models/vae_gaus_model_ld_$(latentdim).pt --samples samples/vis_vae_gaus_ld_$(latentdim)_log.png --latent-dim $(latentdim)

vis-vae-mog:
	python3 vae/vae.py vis --prior mog --model models/vae_mog_model_ld_$(latentdim).pt --samples samples/vis_vae_mog_ld_$(latentdim)_log.png --latent-dim $(latentdim)

vis-vae-vampprior:
	python3 vae/vae.py vis --prior vampprior --model models/vae_vampprior_model_ld_$(latentdim).pt --samples samples/vis_vampprior_ld_$(latentdim)_log.png --latent-dim $(latentdim)

vis-vae-flow:
	python3 vae/vae.py vis --prior flow --model models/vae_flow_model_ld_$(latentdim).pt --samples samples/vis_vae_flow_ld_$(latentdim)_log.png --latent-dim $(latentdim)

vis-vae: vis-vae-gaus vis-vae-mog vis-vae-vampprior vis-vae-flow

sample-vae:
	python3 vae/vae.py sample --model models/vae_gaus_model_ld_$(latentdim).pt --samples samples/sample_vae_gaus_ld_$(latentdim).png --latent-dim $(latentdim)
	python3 vae/vae.py sample --prior mog --model models/vae_mog_model_ld_$(latentdim).pt --samples samples/sample_vae_mog_ld_$(latentdim).png --latent-dim $(latentdim)
	python3 vae/vae.py sample --prior vampprior --model models/vae_vampprior_model_ld_$(latentdim).pt --samples samples/sample_vampprior_ld_$(latentdim).png --latent-dim $(latentdim)
	python3 vae/vae.py sample --prior flow --model models/vae_flow_model_ld_$(latentdim).pt --samples samples/sample_vae_flow_ld_$(latentdim).png --latent-dim $(latentdim)