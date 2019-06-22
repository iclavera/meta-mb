//---------------------------------//
//  This file is part of MuJoCo    //
//  Written by Emo Todorov         //
//  Copyright (C) 2017 Roboti LLC  //
//---------------------------------//


#include "mujoco.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// enable compilation with and without OpenMP support
#if defined(_OPENMP)
    #include <omp.h>
#else
    // omp timer replacement
    #include <chrono>
    double omp_get_wtime(void)
    {
        static std::chrono::system_clock::time_point _start = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - _start;
        return elapsed.count();
    }

    // omp functions used below
    void omp_set_dynamic(int) {}
    void omp_set_num_threads(int) {}
    int omp_get_num_procs(void) {return 1;}
#endif


// gloval variables: internal
const int MAXTHREAD = 64;   // maximum number of threads allowed
const int MAXEPOCH = 100;   // maximum number of epochs
int isforward = 0;          // dynamics mode: forward or inverse
//mjtNum* deriv = 0;          // dynamics derivatives (6*nv*nv):
                            //  dinv/dpos, dinv/dvel, dinv/dacc, dacc/dpos, dacc/dvel, dacc/dfrc


// global variables: user-defined, with defaults
int nthread = 0;            // number of parallel threads (default set later)
int niter = 30;             // fixed number of solver iterations for finite-differencing
int nwarmup = 3;            // center point repetitions to improve warmstart
int nepoch = 20;            // number of timing epochs
int nstep = 500;            // number of simulation steps per epoch
double eps = 1e-6;          // finite-difference epsilon


// worker function for parallel finite-difference computation of derivatives
void worker(const mjModel* m, const mjData* dmain, mjData* d, int id, mjtNum* deriv)
{
    int nv = m->nv;

    // allocate stack space for result at center
    mjMARKSTACK
    mjtNum* center = mj_stackAlloc(d, nv);
    mjtNum* warmstart = mj_stackAlloc(d, nv);

    // prepare static schedule: range of derivative columns to be computed by this thread
    int chunk = (m->nv + nthread-1) / nthread;
    int istart = id * chunk;
    int iend = mjMIN(istart + chunk, m->nv);

    // copy state and control from dmain to thread-specific d
    d->time = dmain->time;
    mju_copy(d->qpos, dmain->qpos, m->nq);
    mju_copy(d->qvel, dmain->qvel, m->nv);
    mju_copy(d->qacc, dmain->qacc, m->nv);
    mju_copy(d->qacc_warmstart, dmain->qacc_warmstart, m->nv);
    mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
    mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
    mju_copy(d->ctrl, dmain->ctrl, m->nu);

    // run full computation at center point (usually faster than copying dmain)
    if( isforward )
    {
        mj_forward(m, d);

        // extra solver iterations to improve warmstart (qacc) at center point
        for( int rep=1; rep<nwarmup; rep++ )
            mj_forwardSkip(m, d, mjSTAGE_VEL, 1);
    }
    else
        mj_inverse(m, d);

    // select output from forward or inverse dynamics
    mjtNum* output = (isforward ? d->qacc : d->qfrc_inverse);

    // save output for center point and warmstart (needed in forward only)
    mju_copy(center, output, nv);
    mju_copy(warmstart, d->qacc_warmstart, nv);

    // select target vector and original vector for force or acceleration derivative
    mjtNum* target = (isforward ? d->qfrc_applied : d->qacc);
    const mjtNum* original = (isforward ? dmain->qfrc_applied : dmain->qacc);

    // finite-difference over force or acceleration: skip = mjSTAGE_VEL
    for( int i=istart; i<iend; i++ )
    {
        // perturb selected target
        target[i] += eps;

        // evaluate dynamics, with center warmstart
        if( isforward )
        {
            mju_copy(d->qacc_warmstart, warmstart, m->nv);
            mj_forwardSkip(m, d, mjSTAGE_VEL, 1);
        }
        else
            mj_inverseSkip(m, d, mjSTAGE_VEL, 1);

        // undo perturbation
        target[i] = original[i];

        // compute column i of derivative 2
        for( int j=0; j<nv; j++ )
            deriv[(3*isforward+2)*nv*nv + i + j*nv] = (output[j] - center[j])/eps;
    }

    // finite-difference over velocity: skip = mjSTAGE_POS
    for( int i=istart; i<iend; i++ )
    {
        // perturb velocity
        d->qvel[i] += eps;

        // evaluate dynamics, with center warmstart
        if( isforward )
        {
            mju_copy(d->qacc_warmstart, warmstart, m->nv);
            mj_forwardSkip(m, d, mjSTAGE_POS, 1);
        }
        else
            mj_inverseSkip(m, d, mjSTAGE_POS, 1);

        // undo perturbation
        d->qvel[i] = dmain->qvel[i];

        // compute column i of derivative 1
        for( int j=0; j<nv; j++ )
            deriv[(3*isforward+1)*nv*nv + i + j*nv] = (output[j] - center[j])/eps;
    }

    // finite-difference over position: skip = mjSTAGE_NONE
    for( int i=istart; i<iend; i++ )
    {
        // get joint id for this dof
        int jid = m->dof_jntid[i];

        // get quaternion address and dof position within quaternion (-1: not in quaternion)
        int quatadr = -1, dofpos = 0;
        if( m->jnt_type[jid]==mjJNT_BALL )
        {
            quatadr = m->jnt_qposadr[jid];
            dofpos = i - m->jnt_dofadr[jid];
        }
        else if( m->jnt_type[jid]==mjJNT_FREE && i>=m->jnt_dofadr[jid]+3 )
        {
            quatadr = m->jnt_qposadr[jid] + 3;
            dofpos = i - m->jnt_dofadr[jid] - 3;
        }

        // apply quaternion or simple perturbation
        if( quatadr>=0 )
        {
            mjtNum angvel[3] = {0,0,0};
            angvel[dofpos] = eps;
            mju_quatIntegrate(d->qpos+quatadr, angvel, 1);
        }
        else
            d->qpos[i] += eps;

        // evaluate dynamics, with center warmstart
        if( isforward )
        {
            mju_copy(d->qacc_warmstart, warmstart, m->nv);
            mj_forwardSkip(m, d, mjSTAGE_NONE, 1);
        }
        else
            mj_inverseSkip(m, d, mjSTAGE_NONE, 1);

        // undo perturbation
        mju_copy(d->qpos, dmain->qpos, m->nq);

        // compute column i of derivative 0
        for( int j=0; j<nv; j++ )
            deriv[(3*isforward+0)*nv*nv + i + j*nv] = (output[j] - center[j])/eps;
    }

    mjFREESTACK
}

// main function
int compute_derivatives(const mjModel* m, const mjData* dmain, mjtNum* deriv)
{
    // default nthread = number of logical cores (usually optimal)
    nthread = omp_get_num_procs();

    // make mjData: main, per-thread
    mjData* d[MAXTHREAD];
    for( int n=0; n<nthread; n++ )
        d[n] = mj_makeData(m);

    // allocate derivatives
    //    deriv = (mjtNum*) mju_malloc(6*sizeof(mjtNum)*m->nv*m->nv); // Ignasi: I commented this line

    // set up OpenMP (if not enabled, this does nothing)
    omp_set_dynamic(0);
    omp_set_num_threads(nthread);

    // save solver options
    int save_iterations = m->opt.iterations;
    mjtNum save_tolerance = m->opt.tolerance;

    // allocate statistics
    int nefc = 0;
    double cputm[MAXEPOCH][2];
    mjtNum error[MAXEPOCH][8];

    // run epochs, collect statistics
    // set solver options for main simulation
    m->opt.iterations = save_iterations;
    m->opt.tolerance = save_tolerance;

    // count number of active constraints
    nefc += dmain->nefc;

    // set solver options for finite differences
    m->opt.iterations = niter;
    m->opt.tolerance = 0;

    // test forward and inverse
    for( isforward=0; isforward<2; isforward++ )
    {
        // start timer
        double starttm = omp_get_wtime();

        // run worker threads in parallel if OpenMP is enabled
        #pragma omp parallel for schedule(static)
        for( int n=0; n<nthread; n++ )
            worker(m, dmain, d[n], n, mjtNum* deriv);

        // record duration in ms
        //        cputm[epoch][isforward] = 1000*(omp_get_wtime() - starttm);

    // check derivatives
    //    checkderiv(m,  d[0], error[epoch]);
    }
    /*
    // compute statistics
    double mcputm[2] = {0,0}, merror[8] = {0,0,0,0,0,0,0,0};
    for( int epoch=0; epoch<nepoch; epoch++ )
    {
        mcputm[0] += cputm[epoch][0];
        mcputm[1] += cputm[epoch][1];

        for( int ie=0; ie<8; ie++ )
            merror[ie] += error[epoch][ie];
    }

    // print sizes, timing, accuracy
    printf("sizes   : nv %d, nefc %d\n\n", m->nv, nefc/nepoch);
    printf("inverse : %.2f ms\n", mcputm[0]/nepoch);
    printf("forward : %.2f ms\n\n", mcputm[1]/nepoch);
    printf("accuracy: log10(residual L1 relnorm)\n");
    printf("------------------------------------\n");
    for( int ie=0; ie<8; ie++ )
        printf("  %s : %.2g\n", accuracy[ie], merror[ie]/nepoch);
    printf("\n");
    */

    // shut down
//    mju_free(deriv);
//    mj_deleteData(dmain);
    for( int n=0; n<nthread; n++ )
        mj_deleteData(d[n]);
//    mj_deleteModel(m);
//    mj_deactivate();
    return 0;
}
