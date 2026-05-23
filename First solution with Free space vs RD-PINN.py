"""
=================================================================
  PINN Pe=10 FINAL v3 — Time-Weighted BC, Pe=10, x∈[-1,26]
  Best combination of v1 accuracy + v2 mass conservation
=================================================================
Most important: x = [-1, 26]

  v1 RESULT:  r=0 trivial. 0% error everywhere. No wall correction.
              Mass decays to 0.37. Neumann BC violated (-1.05).

  v2 RESULT:  r≠0 (wall correction working). Mass ABOVE FS ✓.
              BUT mass exceeds 1.0 at t>1.5 (overcorrection).
              BC target at t=0.1 is 1.047 — way too large.
              No time ramp → W_BC×(1.047)² enormous at t=0.1.

  ROOT CAUSE OF v2 FAILURE:
  |∂c_FS/∂z| at z=1, t=0.1 = 1.047  (plume near wall at z=0.5)
  W_BC=2000 → effective force = 2000×(1.047)²/IC² = 0.54
  This is TOO STRONG at early times → large non-physical r.

  THE FIX — TIME-WEIGHTED BC with Pe=10 ramp:
  At t=0.1: plume at x=1, width σ=0.45. NARROW in z.
  Plume center z=0.5, wall at z=1. Distance = 0.5.
  ∂c_FS/∂z at z=1 is large because c_FS is near z=0.5!
  But this is NOT a wall interaction — the plume is small.
  The TRUE wall interaction begins when σ > 0.5 (t>0.1).

  Use: w(t) = sigmoid((t − T_RAMP) / TAU)
  T_RAMP = 0.5  (plume spreads to wall at σ≈1)
  TAU    = 0.15 (smooth transition)

  w(0.1) = 0.069  (BC almost off — correct, plume narrow)
  w(0.3) = 0.269  (BC partial)
  w(0.5) = 0.500  (BC half weight)
  w(1.0) = 0.965  (BC nearly full)
  w(2.0) = 1.000  (BC full)

  This prevents the t=0.1 overcorrection while still enforcing
  BC at t>0.5 where wall reflection physically matters.
  W_BC = 2000 (strong where it matters: t>0.5)

=================================================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
if device.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

Pe    = 10.0
K     = 30.0
EPS   = 0.05;  EPS2 = EPS**2
Z0    = 0.5
T_MAX = 2.0
X_L   = -1.0;  X_R = 26.0
X_CENTER = (X_L+X_R)/2.0
X_HALF   = (X_R-X_L)/2.0

IC_PEAK = 1.0/(2.0*np.pi*EPS2)
_u_z0   = float(1.0 - np.cosh(K*Z0)/np.cosh(K))

# BC time ramp — tuned for Pe=10
T_RAMP = 0.5    # BC activates when plume σ ≈ 1 (at t≈0.5)
TAU    = 0.15   # smooth transition width

print(f"\n── Parameters ──")
print(f"  Pe={Pe}, ε={EPS}, z₀={Z0}, k={K}, T={T_MAX}")
print(f"  Domain: x∈[{X_L},{X_R}], z∈[-1,1]")
print(f"  IC_PEAK={IC_PEAK:.4f}")
print(f"\n── BC Time Ramp (tuned for Pe=10) ──")
print(f"  w(t) = sigmoid((t-{T_RAMP})/{TAU})")
print(f"  T_RAMP={T_RAMP}: BC activates at t≈{T_RAMP} (σ≈1, plume reaching wall)")
for tv in [0.1,0.3,0.5,0.7,1.0,1.5,2.0]:
    w=1/(1+np.exp(-(tv-T_RAMP)/TAU))
    sigma=np.sqrt(EPS2+2*tv)
    print(f"    t={tv}: w={w:.4f}  σ={sigma:.3f}  "
          f"W_eff={2000*w:.0f}")

_LCK = float(K + np.log1p(np.exp(-2*K)) - np.log(2))

def u_vel(z):
    kz=K*z; akz=kz.abs()
    return 1.0-torch.exp(akz+torch.log1p(torch.exp(-2*akz))-np.log(2)-_LCK)

def c_FS_torch(x,z,t):
    var=EPS2+2.0*t; xc=Pe*_u_z0*t
    r2=(x-xc)**2+(z-Z0)**2
    return torch.exp(-r2/(2.0*var))/(2.0*np.pi*var)

def dc_FS_dx(x,z,t):
    var=EPS2+2.0*t; xc=Pe*_u_z0*t
    return c_FS_torch(x,z,t)*(-(x-xc)/var)

def dc_FS_dz(x,z,t):
    var=EPS2+2.0*t
    return c_FS_torch(x,z,t)*(-(z-Z0)/var)

def c_FS_np(x_np,z_np,t_val):
    var=EPS2+2.0*t_val; xc=Pe*_u_z0*t_val
    r2=(x_np-xc)**2+(z_np-Z0)**2
    return np.exp(-r2/(2.0*var))/(2.0*np.pi*var)

def source_term(x,z,t):
    return Pe*(_u_z0-u_vel(z))*dc_FS_dx(x,z,t)

class PINN(nn.Module):
    def __init__(self,hidden=64,n_layers=6):
        super().__init__()
        layers=[nn.Linear(3,hidden),nn.Tanh()]
        for _ in range(n_layers-1):
            layers+=[nn.Linear(hidden,hidden),nn.Tanh()]
        layers+=[nn.Linear(hidden,1)]
        self.net=nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight,gain=0.1)
                nn.init.zeros_(m.bias)

    def _encode(self,x,z,t):
        return torch.cat([(x-X_CENTER)/X_HALF, z, 2.0*t/T_MAX-1.0],dim=-1)

    def residual(self,x,z,t):
        return self.net(self._encode(x,z,t))

    def c_total(self,x,z,t):
        return c_FS_torch(x,z,t)+self.residual(x,z,t)

model=PINN(hidden=64,n_layers=6).to(device)
n_p=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nNetwork: {n_p:,} params")
with torch.no_grad():
    _x0=torch.zeros(1,1,device=device)
    _z0=torch.full((1,1),Z0,device=device)
    _t0=torch.zeros(1,1,device=device)
    print(f"  Sanity: r={model.residual(_x0,_z0,_t0).item():.2e}  "
          f"c={c_FS_torch(_x0,_z0,_t0).item():.4f}")

def _D(y,x):
    return torch.autograd.grad(y,x,torch.ones_like(y),
                               create_graph=True,retain_graph=True)[0]

def bc_weight(t):
    return torch.sigmoid((t-T_RAMP)/TAU)

def loss_pde(model,x,z,t):
    r=model.residual(x,z,t)
    rt=_D(r,t); rx=_D(r,x); rz=_D(r,z)
    rxx=_D(rx,x); rzz=_D(rz,z)
    S=source_term(x,z,t).detach()
    R=rt+Pe*u_vel(z)*rx-rxx-rzz-S
    return (R**2).mean()/(IC_PEAK**2)

def loss_ic(model,x,z):
    t0=torch.zeros_like(x)
    return (model.residual(x,z,t0)**2).mean()

def loss_neumann(model,x,z,t):
    """
    Time-weighted Neumann BC.
    w(t) = sigmoid((t-T_RAMP)/TAU) with T_RAMP=0.5
    At t=0.1: w=0.07 (near zero — plume not at wall)
    At t=0.5: w=0.50 (half weight)
    At t=1.0: w=0.97 (full weight)
    This prevents overcorrection at early t.
    """
    r=model.residual(x,z,t)
    rz=_D(r,z)
    dFS_dz=dc_FS_dz(x,z,t).detach()
    w=bc_weight(t).detach()
    return (w*(rz+dFS_dz)**2).mean()/(IC_PEAK**2)

def loss_ff(model,x,z,t):
    r=model.residual(x,z,t)
    cFS=c_FS_torch(x,z,t).detach()
    return ((r+cFS)**2).mean()/(IC_PEAK**2)

def _rg(v): return v.requires_grad_(True)
_log_r_sh=np.log((EPS2+2.0*T_MAX/5.0)/EPS2)

def sample_pde(N):
    N_i=int(N*0.40); N_u=N-N_i
    u_i=torch.rand(N_i,1,device=device)
    t_i=(EPS2*torch.exp(u_i*_log_r_sh)-EPS2)/2.0
    t_i=t_i.clamp(1e-6,T_MAX/5.0)
    t_u=torch.rand(N_u,1,device=device)*T_MAX+1e-6
    x_a=torch.rand(N,1,device=device)*(X_R-X_L)+X_L
    z_a=2.0*torch.rand(N,1,device=device)-1.0
    return _rg(x_a),_rg(z_a),_rg(torch.cat([t_i,t_u],0))

def sample_ic(N):
    N_g=int(N*0.70); N_u=N-N_g
    x_g=(torch.randn(N_g,1,device=device)*EPS*3).clamp(X_L+1e-3,X_R-1e-3)
    z_g=(Z0+torch.randn(N_g,1,device=device)*EPS*3).clamp(-1+1e-3,1-1e-3)
    x_u=torch.rand(N_u,1,device=device)*(X_R-X_L)+X_L
    z_u=2.0*torch.rand(N_u,1,device=device)-1.0
    return _rg(torch.cat([x_g,x_u],0)),_rg(torch.cat([z_g,z_u],0))

def sample_bc(N):
    """60% uniform t, 40% concentrated at t>T_RAMP where BC matters."""
    def _w(zv):
        N_late=int(N*0.40); N_uni=N-N_late
        t_l=torch.rand(N_late,1,device=device)*(T_MAX-T_RAMP)+T_RAMP
        t_u=torch.rand(N_uni,1,device=device)*T_MAX+1e-6
        x=_rg(torch.rand(N,1,device=device)*(X_R-X_L)+X_L)
        z=_rg(torch.full((N,1),zv,device=device))
        t=_rg(torch.cat([t_l,t_u],0))
        return x,z,t
    return _w(-1.0),_w(1.0)

def sample_ff(N):
    side=torch.randint(0,2,(N,1),device=device).float()
    x_val=side*X_R+(1-side)*X_L
    x=_rg(x_val)
    z=_rg(2.0*torch.rand(N,1,device=device)-1.0)
    t=_rg(torch.rand(N,1,device=device)*T_MAX+1e-6)
    return x,z,t

W_PDE=1.0; W_IC=10.0; W_BC=2000.0; W_FF=100.0

print(f"\n── Weights ──")
print(f"  W_PDE={W_PDE}, W_IC={W_IC}, W_BC={W_BC} (time-weighted), W_FF={W_FF}")
print(f"  T_RAMP={T_RAMP}: BC activates after plume spreads to walls")
print(f"\n  Effective W_BC at key times:")
for tv in [0.1,0.3,0.5,1.0,2.0]:
    w=1/(1+np.exp(-(tv-T_RAMP)/TAU))
    print(f"    t={tv}: {W_BC*w:.0f}")

N_PDE=10000; N_IC=5000; N_BC=2000; N_FF=1000

def train(model,n_adam=30000,lr=1e-3,log_every=500):
    opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8)
    sch=torch.optim.lr_scheduler.OneCycleLR(
        opt,max_lr=lr,total_steps=n_adam,
        pct_start=0.05,anneal_strategy='cos',
        div_factor=20,final_div_factor=1000)
    hist={k:[] for k in ['tot','pde','ic','bc','ff']}
    t0=time.time()
    for it in range(n_adam):
        opt.zero_grad()
        xi,zi,ti=sample_pde(N_PDE); Lp=loss_pde(model,xi,zi,ti)
        xc,zc=sample_ic(N_IC); Li=loss_ic(model,xc,zc)
        (xl,zl,tl),(xr,zr,tr)=sample_bc(N_BC)
        Lb=loss_neumann(model,xl,zl,tl)+loss_neumann(model,xr,zr,tr)
        xf,zf,tf=sample_ff(N_FF); Lf=loss_ff(model,xf,zf,tf)
        total=W_PDE*Lp+W_IC*Li+W_BC*Lb+W_FF*Lf
        if not torch.isfinite(total):
            print(f"[{it}] NaN—skip"); sch.step(); continue
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
        for k,v in [('tot',total),('pde',Lp),('ic',Li),('bc',Lb),('ff',Lf)]:
            hist[k].append(v.item())
        if it%log_every==0 or it==n_adam-1:
            print(f"  [{it:5d}/{n_adam}]  "
                  f"tot={total.item():.3e}  pde={Lp.item():.3e}  "
                  f"ic={Li.item():.3e}  bc={Lb.item():.3e}  "
                  f"ff={Lf.item():.3e}  lr={sch.get_last_lr()[0]:.2e}")
    print(f"\n  Adam done in {time.time()-t0:.1f}s")
    return hist

print(f"\n══════ TRAINING ══════")
print(f"  Pe=10, x∈[{X_L},{X_R}], T={T_MAX}")
print(f"  Time-weighted BC: T_RAMP={T_RAMP}, TAU={TAU}")
print(f"  Target: mass≈1 for t∈[0,2], centroid=10t, BC≈0\n")
hist=train(model,n_adam=30000,lr=1e-3,log_every=500)

print("\n══════ L-BFGS ══════")
opt_lb=torch.optim.LBFGS(model.parameters(),max_iter=3000,
    tolerance_grad=1e-10,tolerance_change=1e-12,
    history_size=100,line_search_fn='strong_wolfe')
_xi,_zi,_ti=sample_pde(10000); _xc,_zc=sample_ic(5000)
(_xl,_zl,_tl),(r,s,u)=sample_bc(2000); _xr,_zr,_tr=r,s,u
_xf,_zf,_tf=sample_ff(1000)
def closure():
    opt_lb.zero_grad()
    L=(W_PDE*loss_pde(model,_xi,_zi,_ti)+W_IC*loss_ic(model,_xc,_zc)
      +W_BC*(loss_neumann(model,_xl,_zl,_tl)+loss_neumann(model,_xr,_zr,_tr))
      +W_FF*loss_ff(model,_xf,_zf,_tf))
    L.backward(); return L
t0=time.time(); opt_lb.step(closure)
print(f"  L-BFGS done in {time.time()-t0:.1f}s")

model.eval()

def predict_total(x_np,z_np,t_val,batch=50000):
    X,Z=np.meshgrid(x_np,z_np,indexing='ij')
    fx=X.reshape(-1,1).astype(np.float32); fz=Z.reshape(-1,1).astype(np.float32)
    out=np.zeros((fx.shape[0],1),np.float32)
    with torch.no_grad():
        for i in range(0,fx.shape[0],batch):
            xt=torch.tensor(fx[i:i+batch],device=device)
            zt=torch.tensor(fz[i:i+batch],device=device)
            tt=torch.full_like(xt,float(t_val))
            out[i:i+batch]=model.c_total(xt,zt,tt).cpu().numpy()
    return X,Z,out.reshape(X.shape)

def predict_residual(x_np,z_np,t_val,batch=50000):
    X,Z=np.meshgrid(x_np,z_np,indexing='ij')
    fx=X.reshape(-1,1).astype(np.float32); fz=Z.reshape(-1,1).astype(np.float32)
    out=np.zeros((fx.shape[0],1),np.float32)
    with torch.no_grad():
        for i in range(0,fx.shape[0],batch):
            xt=torch.tensor(fx[i:i+batch],device=device)
            zt=torch.tensor(fz[i:i+batch],device=device)
            tt=torch.full_like(xt,float(t_val))
            out[i:i+batch]=model.residual(xt,zt,tt).cpu().numpy()
    return X,Z,out.reshape(X.shape)

print("\n══════ DIAGNOSTICS ══════")
with torch.no_grad():
    _z0_t=torch.full((1,1),Z0,device=device)
    def _ev(tv):
        tt=torch.full((1,1),tv,device=device)
        xc_t=Pe*_u_z0*tv
        xt=torch.full((1,1),xc_t,device=device)
        ct=model.c_total(xt,_z0_t,tt).item()
        rv=model.residual(xt,_z0_t,tt).item()
        cFS=c_FS_torch(xt,_z0_t,tt).item()
        return ct,rv,cFS,xc_t
    pts={tv:_ev(tv) for tv in [0.0,0.1,0.3,0.5,1.0,1.5,2.0]}

# Reference from v1 and v2
v1_err={0.0:0.0,0.1:0.0,0.3:0.0,0.5:0.0,1.0:0.0,1.5:0.0,2.0:0.0}
v2_err={0.0:0.0,0.1:0.4,0.3:1.5,0.5:8.2,1.0:27.3,1.5:40.6,2.0:46.2}

print(f"\nc at plume center (x=10t, z=z₀):")
print(f"  {'t':>5} {'xc':>5} {'c_PINN':>10} {'r':>10} {'c_FS':>10} "
      f"{'err%':>7}  v1%  v2%")
for tv,(ct,rv,cFS,xc_t) in pts.items():
    err=abs(ct-cFS)/max(cFS,1e-10)*100
    flag="✓" if err<5 else "~" if err<15 else "✗"
    print(f"  {tv:5.2f} {xc_t:5.1f} {ct:10.5f} {rv:10.6f} "
          f"{cFS:10.5f} {err:6.1f}%{flag}  "
          f"{v1_err.get(tv,0):.0f}%  {v2_err.get(tv,0):.0f}%")

Nx_m,Nz_m=400,200
x_m=np.linspace(X_L,X_R,Nx_m); z_m=np.linspace(-1,1,Nz_m)
dx=(X_R-X_L)/(Nx_m-1); dz=2.0/(Nz_m-1)

def mass(tv):
    _,_,C=predict_total(x_m,z_m,tv); return float(C.sum()*dx*dz)
def mass_FS(tv):
    X,Z=np.meshgrid(x_m,z_m,indexing='ij')
    return float(c_FS_np(X,Z,max(tv,1e-9)).sum()*dx*dz)
def centroid(tv):
    X,_,C=predict_total(x_m,z_m,tv)
    return float((X*C).sum()/(C.sum()+1e-15))

v1_mass={0.0:1.000,0.1:0.869,0.3:0.716,0.5:0.627,
         1.0:0.496,1.5:0.422,2.0:0.373}
v2_mass={0.0:1.000,0.1:0.885,0.3:0.851,0.5:0.862,
         1.0:0.930,1.5:1.024,2.0:1.129}

print(f"\nMass (target=1.0 for all t):")
print(f"  {'t':>5} {'PINN':>8} {'FS':>8} {'v1':>8} {'v2':>8}  note")
for tt in [0.0,0.1,0.3,0.5,1.0,1.5,2.0]:
    M=mass(tt); MFS=mass_FS(tt)
    note="✓" if abs(M-1)<0.05 else "↑" if M>MFS+0.02 else "~"
    print(f"  {tt:5.2f} {M:8.4f} {MFS:8.4f} "
          f"{v1_mass.get(tt,0):8.4f} {v2_mass.get(tt,0):8.4f}  {note}")

v1_cen={0.1:1.0,0.5:5.0,1.0:10.0,2.0:19.99}
v2_cen={0.1:1.016,0.5:4.700,1.0:8.627,2.0:16.114}

print(f"\nCentroid (theory=10t):")
print(f"  {'t':>5} {'PINN':>8} {'theory':>8} {'err%':>7}  v1%  v2%")
for tt in [0.1,0.3,0.5,1.0,1.5,2.0]:
    xc=centroid(tt); xt=Pe*_u_z0*tt
    err=abs(xc-xt)/max(xt,1)*100
    v1e=abs(v1_cen.get(tt,xt)-xt)/max(xt,1)*100
    v2e=abs(v2_cen.get(tt,xt)-xt)/max(xt,1)*100
    flag="✓" if err<5 else "~" if err<15 else "✗"
    print(f"  {tt:5.2f} {xc:8.3f} {xt:8.1f} {err:6.1f}%{flag}  "
          f"{v1e:.0f}%  {v2e:.0f}%")

print(f"\nNeumann BC ∂c/∂z at z=+1 (target=0):")
with torch.no_grad():
    for tt_c in [0.1,0.3,0.5,1.0,2.0]:
        tt_t=torch.full((1,1),tt_c,device=device)
        xc_t=float(Pe*_u_z0*tt_c)
        _xm_=torch.full((1,1),xc_t,device=device)
        dz_q=1e-4
        zp_=torch.full((1,1),1.0-dz_q,device=device)
        zw_=torch.full((1,1),1.0,device=device)
        val=(model.c_total(_xm_,zp_,tt_t).item()
            -model.c_total(_xm_,zw_,tt_t).item())/(-dz_q)
        w=1/(1+np.exp(-(tt_c-T_RAMP)/TAU))
        print(f"  t={tt_c}: {val:.6f}  w={w:.3f}  "
              f"{'✓' if abs(val)<0.01 else '~' if abs(val)<0.1 else '✗'}")

# ── PLOTS ────────────────────────────────────────────────────────
Nx_p,Nz_p=400,150
xp=np.linspace(X_L,X_R,Nx_p); zp=np.linspace(-1,1,Nz_p)
XP,ZP=np.meshgrid(xp,zp,indexing='ij')
t_snaps=[0.0,0.5,1.0,1.5,2.0]; cm=plt.cm.plasma

plt.figure(figsize=(11,5))
plt.semilogy(hist['tot'],'k',lw=2,label='Total')
plt.semilogy(hist['pde'],'tab:orange',lw=1.5,label='PDE')
plt.semilogy(hist['ic'],'tab:red',lw=1.2,label='IC')
plt.semilogy(hist['bc'],'tab:blue',lw=1.8,
             label=f'BC W={W_BC}, ramp T={T_RAMP}')
plt.semilogy(hist['ff'],'tab:purple',lw=1.2,label='FF')
plt.xlabel('Iteration',fontsize=12); plt.ylabel('Loss',fontsize=12)
plt.title(f'PINN Pe={Pe} v3 — Time-Weighted BC (T_RAMP={T_RAMP})',fontsize=13)
plt.legend(fontsize=10,ncol=2); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

fig,axes=plt.subplots(1,5,figsize=(22,4.2))
for ax,tt in zip(axes,t_snaps):
    _,_,R=predict_residual(xp,zp,tt)
    vmax=max(abs(R).max(),1e-10)
    pcm=ax.pcolormesh(XP,ZP,R,cmap='RdBu_r',shading='auto',
                      vmin=-vmax,vmax=vmax)
    plt.colorbar(pcm,ax=ax,fraction=0.045,pad=0.04)
    ax.set_title(f't={tt}  |r|={abs(R).max():.3g}',fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('z')
fig.suptitle('Residual r(x,z,t) — wall correction growing for t>0.5',
             fontsize=13,y=1.02)
plt.tight_layout(); plt.show()

fig,axes=plt.subplots(1,5,figsize=(22,4.2))
for ax,tt in zip(axes,t_snaps):
    _,_,C=predict_total(xp,zp,tt)
    vmax=max(float(C.max()),1e-8)
    pcm=ax.pcolormesh(XP,ZP,np.clip(C,0,None),cmap='hot',
                      shading='auto',vmin=0,vmax=vmax)
    plt.colorbar(pcm,ax=ax,fraction=0.045,pad=0.04).ax.tick_params(labelsize=8)
    xc_t=Pe*_u_z0*tt
    if X_L<xc_t<X_R:
        ax.axvline(xc_t,color='cyan',ls='--',lw=1.5,alpha=0.8)
    ax.set_title(f't={tt} peak={C.max():.3g}',fontsize=11,fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('z')
fig.suptitle('Physical c(x,z,t) — cyan=centroid theory',fontsize=13,y=1.02)
plt.tight_layout(); plt.show()

for tt_3d in [0.5,1.0,2.0]:
    _,_,Cpl=predict_total(xp,zp,tt_3d)
    fig=plt.figure(figsize=(12,7)); ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(XP,ZP,Cpl,cmap='viridis',linewidth=0,alpha=0.93)
    ax.set_xlabel('x',labelpad=10); ax.set_ylabel('z',labelpad=10)
    ax.set_zlabel('c',labelpad=10)
    ax.set_title(f'3D c(x,z,t={tt_3d}), Pe={Pe}',fontsize=13,pad=15)
    ax.view_init(elev=30,azim=-60)
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]: p.fill=False
    plt.tight_layout(); plt.show()

x_full=np.linspace(X_L,X_R,800)
plt.figure(figsize=(12,5.5))
for j,tt in enumerate(t_snaps):
    _,_,C=predict_total(x_full,np.array([Z0]),tt)
    plt.plot(x_full,C.ravel(),color=cm(j/(len(t_snaps)-.9)),
             lw=2,label=f't={tt} (xc={Pe*_u_z0*tt:.0f})')
plt.xlabel('x',fontsize=13); plt.ylabel(f'c(x,z={Z0},t)',fontsize=13)
plt.title(f'Horizontal profiles at z=z₀, Pe={Pe}',fontsize=13)
plt.legend(fontsize=10); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

z1d=np.linspace(-1,1,400)
plt.figure(figsize=(10,5.5))
for j,tt in enumerate(t_snaps):
    xcv=float(np.clip(Pe*_u_z0*tt,X_L+0.1,X_R-0.1))
    _,_,C=predict_total(np.array([xcv]),z1d,tt)
    plt.plot(z1d,C.ravel(),color=cm(j/(len(t_snaps)-.9)),
             lw=2,label=f't={tt}')
plt.axvline(-1,color='k',ls='--',lw=1,alpha=0.4)
plt.axvline(1,color='k',ls='--',lw=1,alpha=0.4,label='walls z=±1')
plt.axvline(Z0,color='r',ls=':',lw=1.2,alpha=0.7,label=f'z₀={Z0}')
plt.xlabel('z',fontsize=13); plt.ylabel('c(xc,z,t)',fontsize=13)
plt.title(f'Vertical profiles at plume center — Pe={Pe}',fontsize=13)
plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

t_ax=np.linspace(0.01,T_MAX,30)
M_arr=[mass(tt) for tt in t_ax]
M_fs_arr=[mass_FS(tt) for tt in t_ax]
plt.figure(figsize=(9,5))
plt.plot(t_ax,M_arr,'bo-',lw=2,ms=5,label='PINN v3')
plt.plot(t_ax,M_fs_arr,'g--',lw=1.8,label='Free-space (no walls)')
plt.axhline(1.0,color='r',ls='--',lw=2,label='Theory M=1')
plt.xlabel('t',fontsize=13); plt.ylabel('Total mass',fontsize=13)
plt.title(f'Mass conservation — Pe={Pe}, x∈[{X_L},{X_R}]',fontsize=13)
plt.ylim(0,1.4); plt.legend(fontsize=10); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

t_cv=np.linspace(0.05,T_MAX,25)
xc_pinn=[centroid(tt) for tt in t_cv]
xc_th=[Pe*_u_z0*tt for tt in t_cv]
plt.figure(figsize=(9,5))
plt.plot(t_cv,xc_pinn,'bo-',lw=2,ms=5,label='PINN v3')
plt.plot(t_cv,xc_th,'r--',lw=2,label=f'Theory: {Pe*_u_z0:.0f}·t')
plt.xlabel('t',fontsize=13); plt.ylabel('x_c(t)',fontsize=13)
plt.title(f'Centroid trajectory — Pe={Pe}',fontsize=13)
plt.legend(fontsize=10); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,tt in zip(axes,[0.5,1.0,2.0]):
    xc_t=Pe*_u_z0*tt
    x_local=np.linspace(max(X_L,xc_t-5),min(X_R,xc_t+5),400)
    _,_,Cp=predict_total(x_local,np.array([Z0]),tt)
    CFS_z=c_FS_np(x_local[:,np.newaxis],np.array([[Z0]]),tt).ravel()
    ax.plot(x_local,Cp.ravel(),'b-',lw=2,label='PINN v3')
    ax.plot(x_local,CFS_z,'r--',lw=1.8,label='Free-space')
    ax.axvline(xc_t,color='gray',ls=':',lw=1.2,label=f'xc={xc_t:.0f}')
    ax.set_title(f't={tt} (xc={xc_t:.0f})',fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('c(x,z=z₀)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle(f'PINN v3 vs Free-space at z=z₀={Z0}, Pe={Pe}',fontsize=13,y=1.02)
plt.tight_layout(); plt.show()

print("\n══════ FINAL SUMMARY ══════")
print(f"  Pe={Pe}, domain x∈[{X_L},{X_R}], T={T_MAX}")
print(f"  BC ramp: T_RAMP={T_RAMP}, TAU={TAU}, W_BC={W_BC}")
print(f"\n  Point values (plume center):")
for tv,(ct,rv,cFS,xc_t) in pts.items():
    err=abs(ct-cFS)/max(cFS,1e-10)*100
    print(f"    t={tv}: xc={xc_t:.0f}  c={ct:.5f}  r={rv:.6f}  "
          f"c_FS={cFS:.5f}  err={err:.1f}%")
print(f"\n  Mass:")
for tt in [0.0,0.5,1.0,2.0]:
    M=mass(tt)
    print(f"    t={tt}: M={M:.4f}  {'✓' if abs(M-1)<0.05 else '~'}")
print(f"\n  Centroid:")
for tt in [0.5,1.0,2.0]:
    xc=centroid(tt); xt=Pe*_u_z0*tt
    print(f"    t={tt}: xc={xc:.3f}  theory={xt:.0f}  "
          f"err={abs(xc-xt)/max(xt,1)*100:.1f}%")
