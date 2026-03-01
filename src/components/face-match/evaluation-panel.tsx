'use client';

import { useState } from 'react';
import { EvaluationMetrics } from '@/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Play, 
  Download, 
  BarChart3, 
  Target, 
  TrendingUp,
  Activity,
  Loader2
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface EvaluationPanelProps {
  onEvaluate: () => Promise<void>;
  metrics?: EvaluationMetrics;
  isEvaluating?: boolean;
  className?: string;
}

export function EvaluationPanel({
  onEvaluate,
  metrics,
  isEvaluating = false,
  className,
}: EvaluationPanelProps) {
  const [showDetails, setShowDetails] = useState(false);

  const metricItems = metrics
    ? [
        {
          label: 'Rank-1 准确率',
          value: metrics.rank1,
          icon: Target,
          color: 'text-blue-600',
          bgColor: 'bg-blue-100',
        },
        {
          label: 'Rank-5 准确率',
          value: metrics.rank5,
          icon: Target,
          color: 'text-green-600',
          bgColor: 'bg-green-100',
        },
        {
          label: 'Rank-10 准确率',
          value: metrics.rank10,
          icon: Target,
          color: 'text-purple-600',
          bgColor: 'bg-purple-100',
        },
        {
          label: 'mAP',
          value: metrics.mAP,
          icon: TrendingUp,
          color: 'text-orange-600',
          bgColor: 'bg-orange-100',
        },
        {
          label: 'Precision',
          value: metrics.precision,
          icon: Activity,
          color: 'text-cyan-600',
          bgColor: 'bg-cyan-100',
        },
        {
          label: 'Recall',
          value: metrics.recall,
          icon: Activity,
          color: 'text-pink-600',
          bgColor: 'bg-pink-100',
        },
        {
          label: 'F1-Score',
          value: metrics.f1Score,
          icon: BarChart3,
          color: 'text-indigo-600',
          bgColor: 'bg-indigo-100',
        },
      ]
    : [];

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">实验评估</CardTitle>
            <CardDescription>评估模型在测试集上的性能</CardDescription>
          </div>
          <Button
            onClick={onEvaluate}
            disabled={isEvaluating}
            className="gap-2"
          >
            {isEvaluating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                评估中...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                开始评估
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {!metrics ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <BarChart3 className="h-12 w-12 text-muted-foreground/50 mb-4" />
            <p className="text-muted-foreground">
              点击"开始评估"按钮运行评估
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              需要先上传测试数据集
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* 主要指标 */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {metricItems.slice(0, 4).map((item) => (
                <div
                  key={item.label}
                  className="p-3 rounded-lg border bg-card"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <div className={cn('p-1.5 rounded', item.bgColor)}>
                      <item.icon className={cn('h-4 w-4', item.color)} />
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {item.label}
                    </span>
                  </div>
                  <div className="flex items-end gap-1">
                    <span className={cn('text-2xl font-bold', item.color)}>
                      {(item.value * 100).toFixed(1)}
                    </span>
                    <span className="text-sm text-muted-foreground mb-1">%</span>
                  </div>
                  <Progress value={item.value * 100} className="h-1.5 mt-2" />
                </div>
              ))}
            </div>

            {/* 详细指标 */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDetails(!showDetails)}
              className="w-full"
            >
              {showDetails ? '收起详情' : '展开详情'}
            </Button>

            {showDetails && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 pt-2">
                {metricItems.slice(4).map((item) => (
                  <div
                    key={item.label}
                    className="p-3 rounded-lg border bg-card"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <div className={cn('p-1 rounded', item.bgColor)}>
                        <item.icon className={cn('h-3 w-3', item.color)} />
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {item.label}
                      </span>
                    </div>
                    <span className={cn('text-lg font-bold', item.color)}>
                      {(item.value * 100).toFixed(2)}%
                    </span>
                  </div>
                ))}

                {/* 混淆矩阵 */}
                <div className="col-span-2 md:col-span-3 p-3 rounded-lg border bg-card">
                  <p className="text-sm font-medium mb-2">混淆矩阵</p>
                  <div className="grid grid-cols-2 gap-2 text-center text-sm">
                    <div className="p-2 bg-green-100 rounded">
                      <p className="text-xs text-muted-foreground">TP</p>
                      <p className="font-bold text-green-700">{metrics.truePositive}</p>
                    </div>
                    <div className="p-2 bg-red-100 rounded">
                      <p className="text-xs text-muted-foreground">FP</p>
                      <p className="font-bold text-red-700">{metrics.falsePositive}</p>
                    </div>
                    <div className="p-2 bg-orange-100 rounded">
                      <p className="text-xs text-muted-foreground">FN</p>
                      <p className="font-bold text-orange-700">{metrics.falseNegative}</p>
                    </div>
                    <div className="p-2 bg-gray-100 rounded">
                      <p className="text-xs text-muted-foreground">TN</p>
                      <p className="font-bold text-gray-700">{metrics.trueNegative}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
