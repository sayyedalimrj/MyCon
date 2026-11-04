#!/usr/bin/env python3
"""Exercise the pinned LongSplat rasterizer forward AND backward on actual CUDA."""

def main():
    import json,math,torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    assert torch.cuda.is_available()
    device='cuda'
    view=torch.eye(4,device=device)
    projection=torch.zeros((4,4),device=device)
    projection[0,0]=projection[1,1]=1
    projection[3,2]=1
    projection[2,2]=100/(100-.01)
    projection[2,3]=-(100*.01)/(100-.01)
    projection=projection.T.contiguous()
    config=GaussianRasterizationSettings(image_height=32,image_width=32,tanfovx=1.,tanfovy=1.,
        bg=torch.zeros(3,device=device),scale_modifier=1.,viewmatrix=view,
        projmatrix=projection,projmatrix_raw=projection,sh_degree=0,
        campos=torch.zeros(3,device=device),prefiltered=False,debug=False)
    xyz=torch.tensor([[0.,0.,2.]],device=device,requires_grad=True)
    screen=torch.zeros_like(xyz,requires_grad=True)
    colors=torch.tensor([[1.,.2,.1]],device=device,requires_grad=True)
    opacity=torch.tensor([[.8]],device=device,requires_grad=True)
    scale=torch.full((1,3),.1,device=device,requires_grad=True)
    rotation=torch.tensor([[1.,0.,0.,0.]],device=device,requires_grad=True)
    theta=torch.zeros((1,3),device=device,requires_grad=True)
    rho=torch.zeros((1,3),device=device,requires_grad=True)
    image,radii,depth,alpha,touched=GaussianRasterizer(config)(
        means3D=xyz,means2D=screen,colors_precomp=colors,opacities=opacity,
        scales=scale,rotations=rotation,theta=theta,rho=rho)
    assert torch.isfinite(image).all() and image.sum().item()>0
    assert radii.max().item()>0
    image.sum().backward()
    for value in (xyz,screen,colors,opacity,scale,rotation,theta,rho):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    torch.cuda.synchronize()
    print(json.dumps({'rasterizer_forward':'PASS','rasterizer_backward':'PASS',
                      'sm':torch.cuda.get_device_capability(),'training_executed':False}))

if __name__=='__main__':main()
